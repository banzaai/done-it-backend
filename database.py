from collections import defaultdict


class ConversationManager:
    def __init__(self):
        self.conversations = defaultdict(list)

    async def add_message(self, conversation_id, role, content):
        self.conversations[conversation_id].append({"role": role, "content": content})

    async def get_conversation(self, conversation_id):
        return self.conversations[conversation_id]

    async def clear_conversation(self, conversation_id):
        self.conversations[conversation_id] = []
