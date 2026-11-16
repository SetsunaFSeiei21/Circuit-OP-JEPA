import copy
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from final_exam.models.low_rank_decoder_pin import factorize_decoder_pin
from final_exam.shared_decoder_analysis_pin import (
    DecoderMatrixPin, DecoderGroupPin, SharedDecoderCapturePin,
    analyze_group_pin, apply_approximations_pin, decoder_groups_pin,
)


class Branch(nn.Module):
    def __init__(self, width=8, hidden=16, depth=1):
        super().__init__()
        self.decoder = nn.TransformerDecoder(nn.TransformerDecoderLayer(
            width, 2, hidden, dropout=0., batch_first=True), depth)
        self.query = nn.Parameter(torch.randn(1, 2, width))

    def forward(self, memory, padding):
        return self.decoder(self.query.expand(memory.shape[0], -1, -1), memory,
                            memory_key_padding_mask=padding)


class Fixture(nn.Module):
    def __init__(self, branches=3):
        super().__init__()
        self.decoder = nn.Module()
        self.decoder.branches = nn.ModuleList(Branch() for _ in range(branches))

    def forward(self, memory, padding):
        return torch.cat([b(memory, padding) for b in self.decoder.branches], dim=-1)


def test_vertical_stack_shared_down_and_union_rank():
    # Each branch rank1, but their input row spaces span two directions.
    weights = [torch.tensor([[1., 0., 0.]]), torch.tensor([[0., 1., 0.]])]
    specs = [DecoderMatrixPin(i+1, str(i), w, nn.Linear(3,1), 'linear', 'performance')
             for i,w in enumerate(weights)]
    group = DecoderGroupPin('test', 'performance', 1, 'linear2', specs)
    analysis = analyze_group_pin(group, [1,2,10])
    assert analysis['summary']['numerical_rank'] == 2
    assert analysis['summary']['rank_99'] == 2
    first, second, last = analysis['budget_rows']
    assert first['aggregate_shared_weight_error'] == pytest.approx(2**-.5)
    assert first['aggregate_independent_weight_error'] == pytest.approx(0.)
    assert second['shared_rank'] == 2 and second['independent_rank'] == 1
    assert second['aggregate_shared_weight_error'] < 1e-12
    assert second['shared_parameters'] == 2*3+2*1*2
    assert second['independent_parameters'] == 2*(3+1)
    assert last['shared_rank'] == 2
    for actual, expected in zip(analysis['approximations'][2]['shared'], weights):
        torch.testing.assert_close(actual, expected.double())


def test_groups_separate_self_cross_projection_and_branch_order():
    model = Fixture()
    groups = decoder_groups_pin(model)
    assert len(groups) == 10
    assert {g.role for g in groups} == {
        'self_attn.q','self_attn.k','self_attn.v','self_attn.out',
        'multihead_attn.q','multihead_attn.k','multihead_attn.v','multihead_attn.out',
        'linear1','linear2'}
    assert all([s.branch for s in g.matrices] == [1,2,3] for g in groups)
    assert len(decoder_groups_pin(model, scope='ffn')) == 2
    assert len(decoder_groups_pin(model, scope='attention')) == 8
    with pytest.raises(ValueError, match='at least two'):
        decoder_groups_pin(Fixture(1))
    with pytest.raises(ValueError, match='no per-PE'):
        decoder_groups_pin(model, component='op_predictor')


def test_capture_qkv_ffn_padding_rng_and_no_forward_changes():
    torch.manual_seed(17)
    model = Fixture().eval()
    groups = decoder_groups_pin(model)
    memory = torch.randn(2,5,8)
    padding = torch.tensor([[False,False,False,True,True],[False,False,False,False,True]])
    batch = {'device_mask':~padding[:,1:]}
    with torch.no_grad():
        expected = model(memory,padding)
        rng = torch.get_rng_state().clone()
        with SharedDecoderCapturePin(groups, max_tokens=100) as cap:
            cap.set_batch(batch,0,1)
            actual = model(memory,padding)
        captures = cap.finish()
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    assert torch.equal(torch.get_rng_state(), rng)
    assert not any(m._forward_pre_hooks for m in model.modules())
    for group in groups:
        for spec in group.matrices:
            value = captures[spec.path]
            if group.role.endswith('.out'):
                assert value is None
            else:
                expected_tokens = 7 if group.role in ('multihead_attn.k','multihead_attn.v') else 4
                assert value.shape == (spec.weight.shape[1], expected_tokens)
                assert torch.isfinite(value).all()
    analysis = analyze_group_pin(groups[0], [1,2], captures)
    assert all(r['activation_tokens'] > 0 for r in analysis['error_rows'])
    assert all(r['shared_output_error'] is not None for r in analysis['error_rows'])


@pytest.mark.parametrize('factorized', [False,True])
def test_simulation_restores_exactly_even_on_exception(factorized):
    torch.manual_seed(5)
    model = Fixture().eval()
    if factorized:
        for branch in model.decoder.branches:
            factorize_decoder_pin(branch.decoder, attention_rank=2, ffn_rank=4)
    groups = decoder_groups_pin(model)
    analyses = {g.name:analyze_group_pin(g,[1,8]) for g in groups}
    saved = {k:v.clone() for k,v in model.state_dict().items()}
    memory = torch.randn(2,5,8)
    padding = torch.zeros(2,5,dtype=torch.bool)
    with torch.no_grad():
        original = model(memory,padding)
    with pytest.raises(RuntimeError, match='planned exception'):
        with apply_approximations_pin(groups,analyses,1,'shared'):
            altered = model(memory,padding)
            assert not torch.allclose(altered,original)
            raise RuntimeError('planned exception')
    for key,value in model.state_dict().items():
        torch.testing.assert_close(value,saved[key],rtol=0,atol=0)
    with torch.no_grad():
        torch.testing.assert_close(model(memory,padding),original,rtol=0,atol=0)
    # Every projection is exactly representable by rank8 (input width8 except
    # FFN2, whose branch union can exceed8: use independent strategy here).
    stripped = {k:v['approximations'] for k,v in analyses.items()}
    with apply_approximations_pin(groups,stripped,8,'independent'):
        torch.testing.assert_close(model(memory,padding),original,rtol=2e-5,atol=2e-6)
    assert all('forward' not in m.__dict__ for m in model.modules())


def test_parameter_budgets_and_unweighted_fit():
    matrices=[]
    for branch, scale in enumerate((1.,2.,8.)):
        w=torch.zeros(4,8)
        w[:,branch]=scale
        matrices.append(DecoderMatrixPin(branch+1,str(branch),w,nn.Linear(8,4),'linear','performance'))
    group=DecoderGroupPin('ffn','performance',1,'linear2',matrices)
    analysis=analyze_group_pin(group,[1])
    row=analysis['budget_rows'][0]
    assert row['dense_parameters']==96
    assert row['independent_parameters']==36
    assert row['shared_parameters']==20
    # Raw fit spends rank1 on the high-energy branch, not equal normalized energy.
    torch.testing.assert_close(analysis['approximations'][1]['shared'][2],matrices[2].weight.double())
    for bad in ([],[0],[1,1],[True]):
        with pytest.raises(ValueError): analyze_group_pin(group,bad)


def test_capture_lowrank_effective_projections():
    model=Fixture().eval()
    for branch in model.decoder.branches:
        factorize_decoder_pin(branch.decoder,attention_rank=2,ffn_rank=4)
    groups=decoder_groups_pin(model)
    batch={'device_mask':torch.tensor([[True,True,False,False]])}
    memory=torch.randn(1,5,8)
    padding=torch.tensor([[False,False,False,True,True]])
    with SharedDecoderCapturePin(groups,max_tokens=100) as cap, torch.no_grad():
        cap.set_batch(batch,0,1)
        model(memory,padding)
    captures=cap.finish()
    assert all(value is not None for value in captures.values())
    for group in groups:
        if group.role in ('multihead_attn.k','multihead_attn.v'):
            assert all(captures[s.path].shape[1]==3 for s in group.matrices)


def test_failed_preflight_does_not_mutate_any_weight():
    model=Fixture()
    groups=decoder_groups_pin(model,scope='ffn')
    analyses={g.name:analyze_group_pin(g,[1]) for g in groups}
    saved={k:v.clone() for k,v in model.state_dict().items()}
    analyses[groups[-1].name]['approximations'][1]['shared'][-1].fill_(float('nan'))
    with pytest.raises(ValueError,match='Invalid approximation'):
        with apply_approximations_pin(groups,analyses,1): pass
    for key,value in model.state_dict().items():
        torch.testing.assert_close(value,saved[key],rtol=0,atol=0)


def test_wide_op_decoder_capture_uses_owner_mask_and_separate_groups():
    class OPBranch(Branch):
        def forward(self,memory,padding):
            return self.decoder(memory[:,1:],memory,tgt_key_padding_mask=padding[:,1:],
                                memory_key_padding_mask=padding)
    model=Fixture().eval()
    model.predictor=nn.Module()
    model.predictor.head=nn.Module()
    model.predictor.head.branches=nn.ModuleList(OPBranch() for _ in range(3))
    groups=decoder_groups_pin(model,component='both')
    assert len(groups)==20 and len({g.name for g in groups})==20
    selected=[g for g in groups if g.side=='op_predictor']
    memory=torch.randn(2,4,8)
    padding=torch.tensor([[False,False,True,True],[False,False,False,True]])
    batch={'device_mask':torch.ones(2,12,dtype=torch.bool),'owner_mask':~padding[:,1:]}
    with SharedDecoderCapturePin(selected,max_tokens=100) as cap, torch.no_grad():
        cap.set_batch(batch,0,1)
        for branch in model.predictor.head.branches:
            branch(memory,padding)
    captures=cap.finish()
    for group in selected:
        for spec in group.matrices:
            value=captures[spec.path]
            if spec.kind=='functional_out':
                assert value is None
            else:
                expected=5 if group.role in ('multihead_attn.k','multihead_attn.v') else 3
                assert value.shape[1]==expected


def matrix_group(weights):
    specs = [DecoderMatrixPin(i+1, f'branch_{i+1}', w, nn.Linear(w.shape[1],w.shape[0]),
                              'linear', 'performance') for i,w in enumerate(weights)]
    return DecoderGroupPin('test', 'performance', 1, 'linear2', specs)


def test_pairwise_low_rank_does_not_imply_shared_direction():
    weights = [torch.tensor([[1.,0.,0.]]), torch.tensor([[0.,1.,0.]])]
    result = analyze_group_pin(matrix_group(weights), [1,2])
    for row in result['overlap_rows']:
        assert row['actual_rank'] == 1  # Never fill in arbitrary null vectors.
        assert row['overlap'] == pytest.approx(0.)
        assert row['random_expected_overlap'] == pytest.approx(1/3)
        assert row['chance_adjusted_overlap'] == pytest.approx(-.5)
        assert row['mean_angle_deg'] == pytest.approx(90.)
    assert len(result['angle_rows']) == 2
    for row in result['cross_basis_rows']:
        assert row['own_weight_error'] == pytest.approx(0.)
        assert row['cross_weight_error'] == pytest.approx(1.)
        assert row['excess_relative_weight_sse'] == pytest.approx(1.)
        assert row['retained_vs_own'] == pytest.approx(0.)
        assert row['cross_output_error'] is None


def test_pairwise_sign_order_and_rotated_bases_do_not_change_subspace():
    a = torch.tensor([[4.,0.,0.,0.],[0.,2.,0.,0.]],dtype=torch.float64)
    b = torch.tensor([[0.,-5.,0.,0.],[1.,0.,0.,0.]],dtype=torch.float64)
    # Also rotate the equal-singular-value basis within the same row space.
    c = torch.tensor([[1.,1.,0.,0.],[-1.,1.,0.,0.]],dtype=torch.float64)
    result = analyze_group_pin(matrix_group([a,b,c]), [1,2])
    full = [r for r in result['overlap_rows'] if r['requested_rank']==2]
    assert len(full) == 3
    assert all(r['overlap']==pytest.approx(1.) for r in full)
    assert all(r['max_angle_deg']<1e-5 for r in full)
    assert all(r['chance_adjusted_overlap']==pytest.approx(1.) for r in full)
    vectors = [r['absolute_cosines'] for r in result['vector_cosine_rows']
               if (r['branch_a'],r['branch_b'])==(1,2)]
    torch.testing.assert_close(torch.tensor(vectors),torch.tensor([[0.,1.],[1.,0.]]))
    assert all(r['retained_vs_own']==pytest.approx(1.) for r in result['cross_basis_rows']
               if r['requested_rank']==2)
    assert any(r['cutoff_ambiguous_b'] for r in result['overlap_rows']
               if r['requested_rank']==1 and r['branch_b']==3)


def test_cross_basis_is_directional_and_uses_target_activations():
    a, b = torch.diag(torch.tensor([4.,2.,0.])), torch.diag(torch.tensor([1.,3.,0.]))
    group = matrix_group([a,b])
    captures = {'branch_1':torch.tensor([[0.],[1.],[0.]]),
                'branch_2':torch.tensor([[1.],[0.],[0.]])}
    result = analyze_group_pin(group,[1],captures)
    to_b, to_a = result['cross_basis_rows']
    assert (to_b['donor_branch'],to_b['target_branch']) == (1,2)
    assert (to_a['donor_branch'],to_a['target_branch']) == (2,1)
    assert to_b['retained_vs_own'] == pytest.approx(1/9)
    assert to_a['retained_vs_own'] == pytest.approx(1/4)
    for row in (to_a,to_b):
        assert row['cross_output_error'] == pytest.approx(0.)
        assert row['own_output_error'] == pytest.approx(1.)
        assert row['activation_tokens']==1
        assert row['excess_relative_weight_sse']>0


def test_zero_rank_full_space_and_numerical_nulls_are_explicit():
    group = matrix_group([torch.eye(3),torch.eye(3),torch.zeros(3,3)])
    result = analyze_group_pin(group,[1,3,20])
    real = [r for r in result['overlap_rows'] if (r['branch_a'],r['branch_b'])==(1,2)]
    assert real[0]['cutoff_ambiguous_a'] and real[0]['cutoff_ambiguous_b']
    assert all(r['full_input_space'] and r['chance_adjusted_overlap'] is None for r in real[1:])
    null = [r for r in result['overlap_rows'] if r['branch_b']==3]
    assert all(r['actual_rank']==0 and r['overlap'] is None and r['mean_angle_deg'] is None for r in null)
    assert all(r['status']=='no_nonzero_common_rank' for r in null)
    assert not any(r['branch_b']==3 for r in result['angle_rows']+result['vector_cosine_rows'])
    assert all(r['cross_weight_error'] is None for r in result['cross_basis_rows']
               if 3 in (r['donor_branch'],r['target_branch']))
    # Source-float32 noise below the documented numerical threshold is excluded.
    result = analyze_group_pin(matrix_group([torch.diag(torch.tensor([1.,1e-8])),torch.eye(2)]),[2])
    assert result['overlap_rows'][0]['actual_rank']==1


def test_vector_cap_does_not_cap_subspaces_or_perturb_rng_and_weights():
    torch.manual_seed(77)
    weights=[torch.randn(6,8) for _ in range(3)]
    saved=[w.clone() for w in weights]
    group=matrix_group(weights)
    rng=torch.get_rng_state().clone()
    result=analyze_group_pin(group,[2,5],vector_limit=2)
    assert torch.equal(rng,torch.get_rng_state())
    assert all(r['vector_rank_a']==2 and len(r['absolute_cosines'])==2 for r in result['vector_cosine_rows'])
    assert max(r['actual_rank'] for r in result['overlap_rows'])==5
    disabled=analyze_group_pin(group,[2,5],vector_limit=0)
    assert disabled['vector_cosine_rows']==[]
    assert disabled['overlap_rows']==result['overlap_rows']
    for before,after in zip(saved,weights):
        torch.testing.assert_close(before,after,rtol=0,atol=0)
    for bad in (-1,257,True):
        with pytest.raises(ValueError,match='vector_limit'):
            analyze_group_pin(group,[2],vector_limit=bad)


@pytest.mark.skipif(not torch.cuda.is_available(),reason='CUDA unavailable')
def test_pairwise_cuda_matches_cpu():
    torch.manual_seed(21)
    weights=[torch.randn(8,12) for _ in range(3)]
    cpu=analyze_group_pin(matrix_group(weights),[2,4])
    cuda=analyze_group_pin(matrix_group([w.cuda() for w in weights]),[2,4])
    for a,b in zip(cpu['overlap_rows'],cuda['overlap_rows']):
        for key in ('overlap','mean_angle_deg','chance_adjusted_overlap'):
            assert a[key]==pytest.approx(b[key],abs=1e-6)
    for a,b in zip(cpu['cross_basis_rows'],cuda['cross_basis_rows']):
        assert a['cross_weight_error']==pytest.approx(b['cross_weight_error'],abs=1e-8)
