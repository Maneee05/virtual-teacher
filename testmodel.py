import google.generativeai as genai

genai.configure(api_key="AIzaSyBbSSqRPNl9ImFtSOfWjgYqpYfvyNMZBxU")

for model in genai.list_models():
    print(model.name)