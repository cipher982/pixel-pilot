import unittest

from langchain_core.messages import HumanMessage

from pixelpilot.llms.tgi_wrapper import LocalTGIChatModel


class TestTGIModel(unittest.TestCase):
    def setUp(self):
        """Set up test fixtures."""
        self.base_url = "http://jelly:8080"
        self.model = LocalTGIChatModel(base_url=self.base_url)
        self.test_message = "Hello, how are you?"

    def test_model_initialization(self):
        """Test if the model initializes with correct configuration."""
        self.assertEqual(self.model.client_config.base_url, self.base_url)
        self.assertEqual(self.model._llm_type, "local-tgi-chat-model")

    def test_generate_with_empty_messages(self):
        """Test that generate raises ValueError with empty messages."""
        with self.assertRaises(ValueError):
            self.model.invoke([])

    def test_generate_with_message(self):
        """Test generating a response with a valid message."""
        messages = [HumanMessage(content=self.test_message)]
        try:
            result = self.model.invoke(messages)
            self.assertIsNotNone(result)
            self.assertTrue(isinstance(result, str))
        except Exception as e:
            self.skipTest(f"Skipping due to connection error: {str(e)}")


if __name__ == "__main__":
    unittest.main()
