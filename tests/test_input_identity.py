"""Regression tests for physical-model identity and named external-port roles."""
import copy
import unittest

import torch

from device_jepa.baselines import _collate, _make_model, fit_pin_vocabulary
from device_jepa.data import build_manifest, fit_preprocessor, validate_dataset
from device_jepa.model import DeviceEncoder, FrozenPerformanceModel, JEPAModel, ModelConfig, make_decoder
from device_jepa.tensors import collate_samples
from device_jepa.training import encoder_from_metadata, seed_everything


LVT = "sky130_fd_pr__nfet_01v8_lvt"
STANDARD = "sky130_fd_pr__nfet_01v8"


def identity_dataset():
    data = {"schema_version": 1, "metadata": {"synthetic": True}, "feature_spec": {
        "parameter_names": ["W_m", "L_m", "m", "R_ohm"],
        "op_names": ["id_a", "gm_s"], "metric_names": ["gain"],
        "terminal_names": ["d", "g", "s", "b", "p", "n"]}, "samples": []}
    for circuit in ("A", "B"):
        for index in range(8):
            scale = 1 + index / 10 + (2 if circuit == "B" else 0)
            devices = [
                {"device_id": "M1", "device_type": "NMOS", "model_name": LVT.upper(),
                 "parameters": {"W_m": scale * 1e-6, "L_m": .2e-6, "m": 1.},
                 "terminals": {"d": "out", "g": "in", "s": "0", "b": "0"},
                 "op": {"id_a": scale * 1e-5, "gm_s": scale * 1e-4}},
                {"device_id": "M2", "device_type": "NMOS", "model_name": STANDARD,
                 "parameters": {"W_m": scale * 2e-6, "L_m": .3e-6, "m": 1.},
                 "terminals": {"d": "bias", "g": "bias", "s": "0", "b": "0"},
                 "op": {"id_a": 2e-5, "gm_s": 2e-4}},
                {"device_id": "R1", "device_type": "resistor",
                 "parameters": {"R_ohm": 1000. * scale},
                 "terminals": {"p": "out", "n": "bias"}, "op": {}}]
            ports = {"VIN": "in", "VOUT": "out", "VSS": "0"}
            if circuit == "B":
                devices[0]["model_name"] = "heldout_model"
                ports["heldout_port"] = "bias"
            data["samples"].append({"sample_id": f"{circuit}:{index}", "design_id": str(index),
                "circuit_id": circuit, "devices": devices, "ports": ports,
                "performance": {"gain": 10. * scale}})
    return validate_dataset(data)


class InputIdentityTests(unittest.TestCase):
    def setUp(self):
        seed_everything(32)
        self.data = identity_dataset()
        self.manifest = build_manifest(self.data, config={"train_circuits": ["A"],
            "ood_circuits": ["B"], "split_ratios": [.5, .25, .25], "split_seed": 17,
            "finetune_label_fraction": 1.})
        self.prep = fit_preprocessor(self.data, self.manifest, True)
        self.pin_vocab = fit_pin_vocabulary(self.data, self.manifest)
        self.row = copy.deepcopy(self.data["samples"][0])
        self.cfg = ModelConfig(dim=8, heads=2, encoder_layers=2, cross_layers=1,
            ffn_dim=16, decoder_layers=1, decoder_ffn_dim=16, dropout=0., max_devices=64).to_dict()

    def model_batch(self, mode, rows):
        if mode == "jepa_zerosim":
            encoder = encoder_from_metadata(self.cfg, self.prep)
            model = FrozenPerformanceModel(encoder, make_decoder(encoder.cfg, 1)).eval()
            return model, collate_samples(rows, self.prep, include_op=False)
        cfg = dict(self.cfg)
        if "pin" in mode:
            cfg["terminal_bias"] = False
        return (_make_model(mode, cfg, self.prep, self.pin_vocab).eval(),
                _collate(rows, self.prep, mode, cfg, self.pin_vocab))

    def test_train_only_model_and_port_vocabularies(self):
        self.assertEqual(self.prep["input_contract_version"], 3)
        self.assertEqual([self.prep["model_vocab"][k] for k in ("<PAD>", "<UNK>", "<NONE>")], [0, 1, 2])
        self.assertIn(LVT, self.prep["model_vocab"])
        self.assertNotIn(LVT.upper(), self.prep["model_vocab"])
        self.assertNotIn("heldout_model", self.prep["model_vocab"])
        self.assertNotIn("heldout_port", self.prep["port_role_vocab"])
        self.assertIn("vin", self.prep["port_role_vocab"])
        for field in ("model_vocabulary", "port_role_vocabulary"):
            self.assertEqual(self.prep["fit_ids"][field], sorted(self.manifest["train"]))

    def test_model_identity_distinguishes_all_five_modes(self):
        changed = copy.deepcopy(self.row)
        changed["devices"][0]["model_name"] = STANDARD
        for mode in ("zerosim_device", "zerosim_device_op", "zerosim_pin", "zerosim_pin_op", "jepa_zerosim"):
            with self.subTest(mode=mode):
                model, batch = self.model_batch(mode, [self.row, changed])
                self.assertFalse(torch.equal(batch["device_models"][0], batch["device_models"][1]))
                output = model(batch)
                self.assertFalse(torch.equal(output[0], output[1]))
                if mode != "jepa_zerosim":
                    output.square().sum().backward()
                    grad = model.encoder.structure_encoder.model_embedding.weight.grad
                    self.assertGreater(float(grad.abs().sum()), 0.)

    def test_named_port_permutation_distinguishes_all_five_modes(self):
        changed = copy.deepcopy(self.row)
        changed["ports"]["VIN"], changed["ports"]["VOUT"] = changed["ports"]["VOUT"], changed["ports"]["VIN"]
        for mode in ("zerosim_device", "zerosim_device_op", "zerosim_pin", "zerosim_pin_op", "jepa_zerosim"):
            with self.subTest(mode=mode):
                model, batch = self.model_batch(mode, [self.row, changed])
                self.assertFalse(torch.equal(batch["named_port_roles"][0], batch["named_port_roles"][1]))
                output = model(batch)
                self.assertFalse(torch.equal(output[0], output[1]))
                if mode != "jepa_zerosim":
                    output.square().sum().backward()
                    grad = model.encoder.structure_encoder.named_port_projection.weight.grad
                    self.assertGreater(float(grad.abs().sum()), 0.)

    def test_unknown_none_and_padding_and_parent_identity_are_distinct(self):
        unknown = copy.deepcopy(self.row)
        unknown["devices"][0]["model_name"] = "new_model"
        unknown["ports"] = {"new_port": "in"}
        short = copy.deepcopy(self.row)
        short["devices"] = short["devices"][:1]
        batch = collate_samples([unknown, short], self.prep, include_op=False)
        self.assertEqual(batch["device_models"][0].tolist(), [1, self.prep["model_vocab"][STANDARD], 2])
        self.assertEqual(batch["device_models"][1, 1:].tolist(), [0, 0])
        self.assertFalse(batch["named_port_roles"][1, 1:].any())
        roles = batch["named_port_roles"][0, 0].reshape(6, -1)
        self.assertEqual(float(roles[1, 1]), 1.)
        self.assertFalse(roles[:, 0].any())
        _, pin = self.model_batch("zerosim_pin_op", [unknown, short])
        for index, name in enumerate(pin["device_ids"][0]):
            if name.startswith("port:"):
                self.assertEqual(int(pin["device_models"][0, index]), 2)
                self.assertFalse(pin["named_port_roles"][0, index].any())
            else:
                parent = name.split(":")[1]
                parent_index = batch["device_ids"][0].index(parent)
                self.assertEqual(pin["device_models"][0, index], batch["device_models"][0, parent_index])

    def test_multiple_named_ports_on_one_net_are_not_collapsed(self):
        row = copy.deepcopy(self.row)
        row["ports"]["VOUT"] = "in"
        batch = collate_samples([row], self.prep, include_op=False)
        gate = batch["named_port_roles"][0, 0].reshape(6, -1)[1]
        self.assertEqual(float(gate[self.prep["port_role_vocab"]["vin"]]), 1.)
        self.assertEqual(float(gate[self.prep["port_role_vocab"]["vout"]]), 1.)

    def test_spice_identity_case_does_not_change_tensor_inputs(self):
        changed = copy.deepcopy(self.row)
        changed["devices"][0]["model_name"] = LVT.lower()
        changed["ports"] = {name.lower(): net for name, net in changed["ports"].items()}
        for mode in ("zerosim_device", "zerosim_device_op", "zerosim_pin", "zerosim_pin_op", "jepa_zerosim"):
            model, batch = self.model_batch(mode, [self.row, changed])
            for key, value in batch.items():
                if isinstance(value, torch.Tensor):
                    self.assertTrue(torch.equal(value[0], value[1]), (mode, key))
            output = model(batch)
            torch.testing.assert_close(output[0], output[1], rtol=0, atol=0)

    def test_contract_three_requires_metadata_and_input_channels(self):
        missing = copy.deepcopy(self.prep)
        missing.pop("model_vocab")
        with self.assertRaisesRegex(ValueError, "model_vocab"):
            collate_samples([self.row], missing)
        model, batch = self.model_batch("zerosim_device", [self.row])
        del batch["device_models"]
        with self.assertRaisesRegex(ValueError, "model identities"):
            model(batch)
        model, batch = self.model_batch("zerosim_device", [self.row])
        del batch["named_port_roles"]
        with self.assertRaisesRegex(ValueError, "terminal-port"):
            model(batch)

    def test_legacy_contracts_reconstruct_without_new_embeddings(self):
        for contract in (1, 2):
            prep = copy.deepcopy(self.prep)
            prep["input_contract_version"] = contract
            prep.pop("model_vocab")
            prep.pop("port_role_vocab")
            if contract == 1:
                prep.pop("device_category_vocab")
            old = DeviceEncoder(ModelConfig(**self.cfg), len(prep["device_vocab"]), 4, 6,
                contract, len(prep.get("device_category_vocab", {}))).eval()
            restored = encoder_from_metadata(self.cfg, prep).eval()
            restored.load_state_dict(old.state_dict(), strict=True)
            batch = collate_samples([self.row], prep, include_op=False)
            self.assertNotIn("device_models", batch)
            self.assertNotIn("named_port_roles", batch)
            torch.testing.assert_close(old(batch)[0], restored(batch)[0], rtol=0, atol=0)

    def test_legacy_pin_vocabulary_keeps_its_saved_port_key_semantics(self):
        legacy = copy.deepcopy(self.prep)
        legacy["input_contract_version"] = 2
        legacy.pop("model_vocab")
        legacy.pop("port_role_vocab")
        old_vocab = fit_pin_vocabulary(self.data, self.manifest, input_contract_version=2)
        self.assertIn("port:VIN", old_vocab)
        self.assertNotIn("port:vin", old_vocab)
        self.assertIn("port:vin", self.pin_vocab)
        cfg = {**self.cfg, "terminal_bias": False}
        batch = _collate([self.row], legacy, "zerosim_pin", cfg, old_vocab)
        position = batch["device_ids"][0].index("port:VIN")
        self.assertEqual(int(batch["device_types"][0, position]), old_vocab["port:VIN"])
        self.assertTrue(torch.isfinite(_make_model("zerosim_pin", cfg, legacy, old_vocab)(batch)).all())

    def test_new_structure_still_shares_once_with_both_gradients_and_no_op_dependency(self):
        batch = collate_samples([self.row], self.prep)
        encoder = encoder_from_metadata(self.cfg, self.prep)
        jepa = JEPAModel(encoder, 2).eval()
        calls = []
        handle = jepa.structure_encoder.register_forward_hook(lambda *args: calls.append(1))
        jepa(batch)
        handle.remove()
        self.assertEqual(len(calls), 1)
        self.assertIs(jepa.op_encoder.structure_encoder, jepa.structure_encoder)
        for index in (0, 1):
            jepa.zero_grad(set_to_none=True)
            jepa(batch)[index].square().sum().backward()
            self.assertGreater(float(jepa.structure_encoder.model_embedding.weight.grad.abs().sum()), 0.)
            self.assertGreater(float(jepa.structure_encoder.named_port_projection.weight.grad.abs().sum()), 0.)
        clean = encoder(batch)[0]
        poisoned = copy.deepcopy(batch)
        poisoned["op"].fill_(float("nan"))
        poisoned["op_mask"].zero_()
        poisoned["op_state"] = object()
        torch.testing.assert_close(clean, encoder(poisoned)[0], rtol=0, atol=0)


if __name__ == "__main__":
    unittest.main()
