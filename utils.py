import os
import dotenv


class Utils:
    def __init__(self):
        dotenv.load_dotenv()
        self.token = os.getenv("HUGGINGFACEHUB_API_TOKEN")
