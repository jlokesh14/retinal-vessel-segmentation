from django.test import TestCase
from django.urls import reverse


class SegmentationPageTests(TestCase):

    def test_home_page_loads(self):
        response = self.client.get(reverse("home"))
        self.assertEqual(response.status_code, 200)

    def test_home_page_contains_title(self):
        response = self.client.get(reverse("home"))
        self.assertContains(response, "Retinal")
