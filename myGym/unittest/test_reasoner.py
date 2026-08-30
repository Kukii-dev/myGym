"""Unit tests for the Reasoner module."""

import os
import sys
import unittest

import torch

from myGym.reasoner.model import (
    HumanPoseEncoder,
    ObjectSetEncoder,
    LanguageEncoder,
    CrossAttentionFusion,
    ObjectGroundingHead,
    ActionSequencerHead,
    ReasonerModel,
)
from myGym.reasoner.dataset import ReasonerDataset, Vocabulary, PropertyEmbedder


SAMPLE_DATA = os.path.join(os.path.dirname(__file__), "..", "reasoner", "sample_dataset.json")
BATCH = 2
K = 18       # joints
N = 4        # objects
T = 8        # language tokens
S = 6        # action tokens
D = 64       # model dim


class TestHumanPoseEncoder(unittest.TestCase):
    def test_output_shape(self):
        enc = HumanPoseEncoder(num_joints=K, out_dim=D)
        pose = torch.randn(BATCH, K, 3)
        out = enc(pose)
        self.assertEqual(out.shape, (BATCH, D))


class TestObjectSetEncoder(unittest.TestCase):
    def test_output_shape(self):
        enc = ObjectSetEncoder(obj_input_dim=10, hidden_dim=D, out_dim=D, n_heads=2)
        objects = torch.randn(BATCH, N, 10)
        out = enc(objects)
        self.assertEqual(out.shape, (BATCH, N, D))

    def test_with_mask(self):
        enc = ObjectSetEncoder(obj_input_dim=10, hidden_dim=D, out_dim=D, n_heads=2)
        objects = torch.randn(BATCH, N, 10)
        mask = torch.zeros(BATCH, N, dtype=torch.bool)
        mask[:, -1] = True
        out = enc(objects, mask=mask)
        self.assertEqual(out.shape, (BATCH, N, D))


class TestLanguageEncoder(unittest.TestCase):
    def test_with_token_ids(self):
        enc = LanguageEncoder(vocab_size=100, embed_dim=D, n_heads=2)
        tokens = torch.randint(0, 100, (BATCH, T))
        seq_out, sent_vec = enc(tokens)
        self.assertEqual(seq_out.shape, (BATCH, T, D))
        self.assertEqual(sent_vec.shape, (BATCH, D))

    def test_with_embeddings(self):
        enc = LanguageEncoder(vocab_size=0, embed_dim=D, n_heads=2)
        embs = torch.randn(BATCH, T, D)
        seq_out, sent_vec = enc(embs)
        self.assertEqual(seq_out.shape, (BATCH, T, D))
        self.assertEqual(sent_vec.shape, (BATCH, D))


class TestCrossAttentionFusion(unittest.TestCase):
    def test_output_shape(self):
        fusion = CrossAttentionFusion(d_model=D, n_heads=2)
        sent = torch.randn(BATCH, D)
        pose = torch.randn(BATCH, D)
        obj_emb = torch.randn(BATCH, N, D)
        out = fusion(sent, pose, obj_emb)
        self.assertEqual(out.shape, (BATCH, D))


class TestObjectGroundingHead(unittest.TestCase):
    def test_output_shape(self):
        head = ObjectGroundingHead(d_model=D)
        ctx = torch.randn(BATCH, D)
        obj_emb = torch.randn(BATCH, N, D)
        logits = head(ctx, obj_emb)
        self.assertEqual(logits.shape, (BATCH, N))


class TestActionSequencerHead(unittest.TestCase):
    def test_teacher_forcing(self):
        head = ActionSequencerHead(action_vocab_size=10, d_model=D, n_heads=2)
        ctx = torch.randn(BATCH, D)
        tgt = torch.randint(0, 10, (BATCH, S))
        logits = head(ctx, tgt)
        self.assertEqual(logits.shape, (BATCH, S, 10))

    def test_generate(self):
        head = ActionSequencerHead(action_vocab_size=10, d_model=D, n_heads=2)
        ctx = torch.randn(BATCH, D)
        seq = head.generate(ctx, start_token_id=1, end_token_id=2, max_len=10)
        self.assertEqual(seq.shape[0], BATCH)
        self.assertGreaterEqual(seq.shape[1], 1)


class TestReasonerModel(unittest.TestCase):
    def setUp(self):
        self.model = ReasonerModel(
            num_joints=K, obj_input_dim=10, action_vocab_size=10,
            d_model=D, n_heads=2, vocab_size=100,
            max_objects=N, max_action_len=S, max_seq_len=T,
        )

    def test_forward(self):
        out = self.model(
            pose=torch.randn(BATCH, K, 3),
            objects=torch.randn(BATCH, N, 10),
            lang_tokens=torch.randint(0, 100, (BATCH, T)),
            action_tokens=torch.randint(0, 10, (BATCH, S)),
        )
        self.assertEqual(out["obj_logits"].shape, (BATCH, N))
        self.assertEqual(out["action_logits"].shape, (BATCH, S, 10))
        self.assertEqual(out["context"].shape, (BATCH, D))

    def test_predict(self):
        result = self.model.predict(
            pose=torch.randn(BATCH, K, 3),
            objects=torch.randn(BATCH, N, 10),
            lang_tokens=torch.randint(0, 100, (BATCH, T)),
        )
        self.assertEqual(result["obj_idx"].shape, (BATCH,))
        self.assertEqual(result["action_seq"].shape[0], BATCH)

    def test_backward(self):
        out = self.model(
            pose=torch.randn(BATCH, K, 3),
            objects=torch.randn(BATCH, N, 10),
            lang_tokens=torch.randint(0, 100, (BATCH, T)),
            action_tokens=torch.randint(0, 10, (BATCH, S)),
        )
        loss = out["obj_logits"].sum() + out["action_logits"].sum()
        loss.backward()
        grads = [p.grad is not None for p in self.model.parameters() if p.requires_grad]
        self.assertTrue(all(grads))


class TestVocabulary(unittest.TestCase):
    def test_encode_decode(self):
        v = Vocabulary(["pick", "up", "the", "red", "cube"])
        ids = v.encode(["pick", "up", "the", "red", "cube"])
        tokens = v.decode(ids)
        self.assertEqual(tokens, ["pick", "up", "the", "red", "cube"])

    def test_special_tokens(self):
        v = Vocabulary()
        self.assertEqual(v.pad_id, 0)
        self.assertEqual(v.start_id, 1)
        self.assertEqual(v.end_id, 2)

    def test_unknown_token(self):
        v = Vocabulary(["hello"])
        ids = v.encode(["hello", "world"])
        self.assertEqual(ids[1], v.token2id["<UNK>"])


class TestPropertyEmbedder(unittest.TestCase):
    def test_encode_object(self):
        pe = PropertyEmbedder()
        objects = [[
            {"position": [1, 2, 3], "orientation": [0, 0, 0],
             "size": "small", "color": "red", "shape": "cube", "texture": "smooth"}
        ]]
        pe.fit(objects)
        vec = pe.encode_object(objects[0][0])
        self.assertEqual(len(vec), 10)
        self.assertEqual(vec[:3], [1, 2, 3])


class TestReasonerDataset(unittest.TestCase):
    @unittest.skipUnless(os.path.exists(SAMPLE_DATA), "Sample dataset not found")
    def test_load_and_getitem(self):
        ds = ReasonerDataset(SAMPLE_DATA, max_objects=8, max_lang_tokens=16, max_action_len=10)
        self.assertGreater(len(ds), 0)
        item = ds[0]
        self.assertEqual(item["pose"].shape, (18, 3))
        self.assertEqual(item["objects"].shape, (8, 10))
        self.assertEqual(item["lang_tokens"].shape, (16,))
        self.assertEqual(item["action_input"].shape, (10,))

    @unittest.skipUnless(os.path.exists(SAMPLE_DATA), "Sample dataset not found")
    def test_dataloader(self):
        ds = ReasonerDataset(SAMPLE_DATA, max_objects=8, max_lang_tokens=16, max_action_len=10)
        loader = torch.utils.data.DataLoader(ds, batch_size=2)
        batch = next(iter(loader))
        self.assertEqual(batch["pose"].shape[0], 2)


if __name__ == "__main__":
    unittest.main()
