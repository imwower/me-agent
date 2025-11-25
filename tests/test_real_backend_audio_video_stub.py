import unittest

from me_ext.backends.real_backend import RealEmbeddingBackend
from me_core.types import AudioRef, VideoRef


class RealBackendAudioVideoStubTest(unittest.TestCase):
    def test_embed_audio_video(self) -> None:
        backend = RealEmbeddingBackend({"use_stub": True, "dim": 16})
        audio_vecs = backend.embed_audio([AudioRef(path="a.wav")])
        video_vecs = backend.embed_video([VideoRef(path="b.mp4")])
        self.assertEqual(len(audio_vecs), 1)
        self.assertEqual(len(video_vecs), 1)
        self.assertEqual(len(audio_vecs[0]), 16)
        self.assertEqual(len(video_vecs[0]), 16)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
