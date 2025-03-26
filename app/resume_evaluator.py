import requests
from config import Config

class ResumeEvaluator:
    def __init__(self):
        self.api_key = Config.API_KEY
        self.api_url = "https://api.groq.com/openai/v1/chat/completions"  # Updated URL

    def create_prompt(self, resume_text, target_role):
        """
        Create a prompt for resume evaluation
        """
        system_prompt = """You are an expert resume evaluator and career advisor with deep knowledge of ATS 
        (Applicant Tracking Systems), recruitment best practices, and industry-specific hiring trends. 
        Your task is to analyze a candidate's resume against a specific job role and provide a structured 
        assessment, feedback, and actionable improvement suggestions."""
        
        user_prompt = f"""Analyze the following resume data and generate a structured evaluation for the {target_role} role specifically.
Provide an ATS score breakdown across different categories, followed by improvement suggestions.
Keep the breakdown short and structured.
> Do not provide explanations within the score breakdown.
> Ensure the final recommendations are concise but detailed, using examples where applicable.
> The response must strictly be in JSON format, containing all the below information.

### ATS Score Breakdown
[Only provide the scores, no explanations.]

  **Role Match**: [Score out of 100] \n
  **Experience & Achievements**: [Score out of 100] \n
  **Skills Match**: [Score out of 100] \n
  **Education Fit**: [Score out of 100] \n
  **Readability & Grammar**: [Score out of 100] \n
  **Formatting & ATS Compliance**: [Score out of 100] \n
  **Keyword Density & Buzzword Balance**: [Score out of 100] \n
  **Action Verbs & Impactful Language**: [Score out of 100] \n

### Final Recommendations
[Summarize key improvement areas with direct examples and specific action points.]

> **Major improvement area**: [Brief explanation with an example]\n
> **Another key issue**: [How to fix it]\n
> [Additional concise tips if applicable]

### Final ATS Score:
[Only show the calculated score, not the formula.]
> Score: [Calculated value]
[Formula: total_score = (
    (role_match * 25 / 100) +
    (experience * 20 / 100) +
    (skills * 15 / 100) +
    (education * 10 / 100) +
    (grammar * 10 / 100) +
    (formatting * 7 / 100) +
    (keyword_density * 5 / 100) +
    (action_verbs * 5 / 100)
)]

Resume Data:
''' {resume_text} '''

"""
        
        return system_prompt, user_prompt

    def evaluate_resume(self, resume_text, target_role):
        """
        Evaluate a resume for a specific role
        """
        system_prompt, user_prompt = self.create_prompt(resume_text, target_role)
        
        try:
            print(f"Using API URL: {self.api_url}")
            
            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json"
            }
            
            payload = {
                "model": "llama-3.3-70b-versatile",
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                "temperature": 0.5,
                "max_tokens": 2000
            }
            
            response = requests.post(
                self.api_url,
                headers=headers,
                json=payload
            )
            
            print(f"Response status: {response.status_code}")
            if response.status_code != 200:
                print(f"Error response: {response.text}")
                
            if response.status_code == 200:
                result = response.json()
                return result.get("choices", [{}])[0].get("message", {}).get("content", "")
            else:
                raise Exception(f"API Error: {response.status_code}, {response.text}")
                
        except Exception as e:
            print(f"Exception details: {str(e)}")
            raise Exception(f"Error evaluating resume: {str(e)}")