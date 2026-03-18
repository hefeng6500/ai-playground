from openai import OpenAI

client = OpenAI(
)

completion = client.chat.completions.create(
    model="glm-4.7",
    messages=[
        {"role": "system", "content": "You are a smart and creative novelist"},
        {
            "role": "user",
            "content": "Please write a short fairy tale story as a fairy tale master",
        },
    ],
)

print(completion.choices[0].message.content)