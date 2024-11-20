import unittest
from flask_app.app import app


class FlaskAppTests(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        # Create a test client from the Flask app
        cls.client = app.test_client()

    def test_home_page(self):
        # Test if the home page loads successfully
        response = self.client.get("/")
        self.assertEqual(response.status_code, 200)
        # Check for the title of the app
        self.assertIn(b"<title>Sentiment Analysis</title>", response.data)
        # Check for the heading
        self.assertIn(b"<h1>Sentiment Analysis App</h1>", response.data)

    def test_predict_page(self):
        # Test the predict page with sample input
        response = self.client.post("/predict", data=dict(text="I love this!"))
        self.assertEqual(response.status_code, 200)
        # Ensure the response contains either "Happy" or "Sad"
        self.assertTrue(
            b"Happy" in response.data or b"Sad" in response.data,
            "Response should contain either 'Happy' or 'Sad'",
        )

    def test_predict_with_empty_input(self):
        # Test the predict page with an empty input
        response = self.client.post("/predict", data=dict(text=""))
        self.assertEqual(response.status_code, 200)
        # Check for proper handling of empty input (modify based on app's behavior)
        self.assertIn(
            b"Type your text here...",
            response.data,
            "Response should prompt the user to type text",
        )


if __name__ == "__main__":
    unittest.main()
