import logging
import unittest

from me_core.perception import (
    AudioEncoderStub,
    ImageEncoderStub,
    TextEncoderStub,
    VideoEncoderStub,
    encode_multimodal,
)
from me_core.types import MultiModalInput

# 为测试输出配置基础日志，便于观察编码过程
logging.basicConfig(level=logging.INFO)


class EncoderDeterminismTestCase(unittest.TestCase):
    """各模态编码桩的一致性与区分度测试。"""

    def test_text_encoder_deterministic_and_distinct(self) -> None:
        encoder = TextEncoderStub()
        v1 = encoder.encode("你好，世界")
        v2 = encoder.encode("你好，世界")
        v3 = encoder.encode("另一段文本")

        # 同一输入多次编码结果应完全相同
        self.assertEqual(v1, v2)
        # 不同输入一般情况下应产生不同向量
        self.assertNotEqual(v1, v3)
        # 向量长度固定
        self.assertEqual(len(v1), 16)

    def test_image_encoder_deterministic_and_distinct(self) -> None:
        encoder = ImageEncoderStub()
        meta1 = {"filename": "a.png", "label": "cat"}
        meta2 = {"filename": "b.png", "label": "dog"}

        v1 = encoder.encode(meta1)
        v2 = encoder.encode(meta1)
        v3 = encoder.encode(meta2)

        self.assertEqual(v1, v2)
        self.assertNotEqual(v1, v3)
        self.assertEqual(len(v1), 16)

    def test_audio_video_encoder_basic(self) -> None:
        audio_encoder = AudioEncoderStub()
        video_encoder = VideoEncoderStub()

        audio_v1 = audio_encoder.encode({"filename": "a.wav"})
        audio_v2 = audio_encoder.encode({"filename": "a.wav"})
        audio_v3 = audio_encoder.encode({"filename": "b.wav"})

        video_v1 = video_encoder.encode({"filename": "a.mp4"})
        video_v2 = video_encoder.encode({"filename": "a.mp4"})
        video_v3 = video_encoder.encode({"filename": "b.mp4"})

        self.assertEqual(audio_v1, audio_v2)
        self.assertNotEqual(audio_v1, audio_v3)
        self.assertEqual(len(audio_v1), 16)

        self.assertEqual(video_v1, video_v2)
        self.assertNotEqual(video_v1, video_v3)
        self.assertEqual(len(video_v1), 16)


class MultiModalEncodeTestCase(unittest.TestCase):
    """多模态输入与组合编码测试。"""

    def test_encode_multimodal_basic(self) -> None:
        """MultiModalInput 应能被 encode_multimodal 正常处理。"""

        m = MultiModalInput(
            text="测试多模态编码",
            image_meta={"filename": "img.png", "label": "test"},
            audio_meta={"filename": "audio.wav"},
            video_meta={"filename": "video.mp4"},
        )

        result = encode_multimodal(m)

        # 应包含各模态的编码结果
        self.assertIn("text", result)
        self.assertIn("image", result)
        self.assertIn("audio", result)
        self.assertIn("video", result)

        for key, vec in result.items():
            with self.subTest(modality=key):
                self.assertIsInstance(vec, list)
                self.assertEqual(len(vec), 16)
                # 向量元素应为 float
                self.assertTrue(all(isinstance(x, float) for x in vec))


if __name__ == "__main__":
    unittest.main()

