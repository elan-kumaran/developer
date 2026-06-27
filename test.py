from dotenv import load_dotenv
import anthropic

load_dotenv()  # loads ANTHROPIC_API_KEY from the .env file into the environment

client = anthropic.Anthropic()  # picks up ANTHROPIC_API_KEY from the environment

response = client.messages.create(
    model="claude-haiku-4-5",
    max_tokens=100,
    messages=[
        {"role": "user", "content": "What is the capital of France?"}
    ],
)

# response.content is a list of content blocks — print the text ones
for block in response.content:
    print(response.content)
    if block.type == "text":
        print(block.text)

