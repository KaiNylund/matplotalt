import os
from openai import OpenAI, AzureOpenAI

# model="TURBO" model="gpt-4-vision-preview"
def get_openai_vision_response(api_key, prompt, base64_img, model="gpt-4-vision-preview", use_azure=False,
                               max_tokens=300, return_full_response=False):
    if use_azure:
        client = AzureOpenAI(api_key=api_key,
                             api_version=os.getenv("OPENAI_API_VERSION"),
                             base_url=os.getenv("AZURE_BASE_URL"))
                             #azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"))
    else:
        client = OpenAI(api_key=api_key)
    response = client.chat.completions.create(
        model=model,
        messages=[
            {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": prompt},
                {
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/jpeg;base64,{base64_img}",
                },
                },
            ],
            }
        ],
        max_tokens=max_tokens
    )
    if return_full_response:
        return response
    return response.choices[0].message.content


def get_huggingface_model_response():
    pass