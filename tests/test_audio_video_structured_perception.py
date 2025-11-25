import json
import unittest

from me_core.perception import AudioPerception, VideoPerception, StructuredPerception


class AudioVideoStructuredPerceptionTest(unittest.TestCase):
    def test_audio_perception_stub(self) -> None:
        p = AudioPerception()
        events = p.perceive("path/to/audio.wav")
        self.assertEqual(events[0].modality, "audio")
        self.assertEqual(events[0].payload.get("path"), "path/to/audio.wav")

    def test_video_perception_stub(self) -> None:
        p = VideoPerception()
        events = p.perceive({"path": "path/to/video.mp4"})
        self.assertEqual(events[0].modality, "video")
        self.assertIn("video_path", events[0].payload)

    def test_structured_perception_json(self) -> None:
        p = StructuredPerception()
        events = p.perceive(json.dumps({"cpu": 0.9, "mem": 0.8}))
        self.assertEqual(events[0].modality, "structured")
        data = events[0].payload.get("data")
        self.assertEqual(data["cpu"], 0.9)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
