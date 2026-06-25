from openai import OpenAI

def generate_explanation(client, row):
    prompt = f"""
    SAP expert analysis:

    Transport ID: {row['transport_id']}
    Module: {row['module']}
    Risk: {row['Predicted Risk']}

    Explain:
    - Why risky
    - Business impact
    - Recommendations
    """

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}]
    )

    return response.choices[0].message.content