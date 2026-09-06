import unittest

from chair_ans import CHAIR


class ChairObjectEventTest(unittest.TestCase):
    def make_chair(self):
        chair = CHAIR.__new__(CHAIR)
        chair.mscoco_objects = ["fire hydrant", "hydrant", "dog", "chair", "seat"]
        chair.inverse_synonym_dict = {
            "fire hydrant": "fire hydrant", "hydrant": "fire hydrant",
            "dog": "dog", "chair": "chair", "seat": "chair",
        }
        chair.double_word_dict = {"fire hydrant": "fire hydrant"}
        return chair

    def test_mentions_preserve_chair_categories_and_character_spans(self):
        mentions = self.make_chair().caption_to_object_mentions("A fire hydrant and a dog.")
        self.assertEqual([item["object_category"] for item in mentions], ["fire hydrant", "dog"])
        self.assertEqual([(item["char_start"], item["char_end"]) for item in mentions], [(2, 14), (21, 24)])
        self.assertEqual([item["object_text"] for item in mentions], ["fire hydrant", "dog"])


if __name__ == "__main__":
    unittest.main()
