import json
import random

DATA_FILE = 'personachat_self_original.json'

class PersonaChat():
    def __init__(self, file_path):
        with open(DATA_FILE, "r", encoding="utf-8") as read_file:
            self.data = json.load(read_file)

        print('Read {} training examples and {} validation examples'.format(
            len(self.data['train']), len(self.data['valid'])
        ))

    def get_conversation(self, index=None, split='train'):

        split = self.data[split]
        if not index:
            index = random.randint(0, len(split))
        sample = split[index]

        persona = sample['personality']
        utterances = sample['utterances']

        conversation = utterances[-1]['history'] + [utterances[-1]['candidates'][-1]]

        print('PERSONA {}\n{}'.format(
            '='*33, '\n'.join(persona))
        )
        print('CONVERSATION {}\n- {}'.format(
            '='*33, '\n- '.join(conversation))
        )

if __name__ == "__main__":

    dataset = PersonaChat(DATA_FILE)
    dataset.get_conversation()