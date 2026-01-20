from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

class AIAgent:
    def __init__(self, api_key):
        if not api_key:
            raise ValueError("API Key is missing.")
            
        self.llm = ChatGoogleGenerativeAI(
            model="gemini-2.5-flash",
            google_api_key=api_key,
            temperature=0.3
        )

    def ask_agent(self, context_summary, user_question):
        """
        Chains the context and question to get a strategic answer.
        """
        prompt = ChatPromptTemplate.from_template("""
        You are a Data Strategy Consultant. 
        
        Here is the current Customer Segmentation Analysis:
        {context}
        
        User's Question: {question}
        
        Instructions:
        1. Answer based on the provided segment data.
        2. If relevant, suggest marketing actions (e.g., "Upsell to Segment 2").
        3. Be concise and professional.
        """)
        
        chain = prompt | self.llm | StrOutputParser()
        return chain.invoke({
            "context": context_summary,
            "question": user_question
        })