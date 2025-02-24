import os
import re
from datetime import datetime, timedelta
from typing import Dict, Optional
import phonenumbers
from email_validator import validate_email, EmailNotValidError
import dateparser

from langchain_google_genai import GoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import TextLoader
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain.tools import Tool
from langchain.agents import initialize_agent, AgentType
from langchain_core.documents import Document
from dotenv import load_dotenv


os.environ["GOOGLE_API_KEY"] = "AIzaSyDinDQGEnpNg65GAn95pbBiOzcOu9jMQP8"
llm = GoogleGenerativeAI(model="gemini-pro", temperature=0.7)

# Loading Hospital Information Document
HOSPITAL_INFO = """
Surya Hospital - Comprehensive Healthcare Guide

Location & Contact:
- Address: 123 Health Avenue, Medical District
- Phone: +1-555-SURYA-HEALTH
- Emergency: +1-555-911-SURYA
- Email: info@suryahospital.com
- Website: www.suryahospital.com

Departments & Services:
1. General Medicine
   - Regular check-ups
   - Preventive care
   - Chronic disease management

2. Cardiology
   - Heart disease treatment
   - ECG and stress tests
   - Cardiac rehabilitation

3. Orthopedics
   - Joint replacement
   - Sports injuries
   - Physiotherapy

4. Pediatrics
   - Child healthcare
   - Vaccinations
   - Growth monitoring

5. Neurology
   - Brain and nerve disorders
   - Headache clinic
   - Stroke management

6. Emergency Care (24/7)
   - Immediate medical attention
   - Trauma care
   - Critical care

Working Hours:
- Monday to Friday: 8:00 AM - 8:00 PM
- Saturday: 9:00 AM - 5:00 PM
- Sunday: Emergency Services Only

Appointment Guidelines:
- Book appointments at least 24 hours in advance
- Bring valid ID and insurance information
- Arrive 15 minutes before appointment time
- Bring previous medical records if available
- Cancel appointments at least 12 hours before scheduled time

Insurance & Payment:
- Accepts all major insurance providers
- Various payment options available
- Financial assistance programs for eligible patients

About Us:
Surya Hospital is committed to providing exceptional healthcare services with compassion and expertise.
"""

class HospitalChatbot:
    def __init__(self):
        self.hospital_name = "Surya Hospital"
        self.user_info = {}
        self.appointments = []
        self.setup_knowledge_base()
        self.setup_agent()

    def setup_knowledge_base(self):
        """Loads hospital information into FAISS for document-based Q&A."""
        try:
            document = [Document(page_content=HOSPITAL_INFO)]
            text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
            chunks = text_splitter.split_documents(document)

            embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
            self.vectorstore = FAISS.from_documents(chunks, embeddings)
            self.qa_chain = RetrievalQA.from_chain_type(
                llm=llm,
                chain_type="stuff",
                retriever=self.vectorstore.as_retriever()
            )
        except Exception as e:
            raise RuntimeError(f"Error setting up knowledge base: {str(e)}")

    def validate_phone(self, phone: str) -> bool:
        """Validates phone number format."""
        try:
            parsed = phonenumbers.parse(phone, "US")
            return phonenumbers.is_valid_number(parsed)
        except phonenumbers.NumberParseException:
            return False

    def validate_email(self, email: str) -> bool:
        """Validates email format."""
        try:
            validate_email(email)
            return True
        except EmailNotValidError:
            return False

    def parse_date(self, date_str: str) -> Optional[str]:
        """Parses natural language date (e.g., 'next Monday') into YYYY-MM-DD."""
        date = dateparser.parse(date_str, settings={'PREFER_DATES_FROM': 'future'})
        if date:
            return date.strftime('%Y-%m-%d')
        return None

    def collect_user_info(self, name: str, phone: str, email: str) -> Dict:
        """Collects and validates user information."""
        if not self.validate_phone(phone):
            raise ValueError("Invalid phone number format")
        if not self.validate_email(email):
            raise ValueError("Invalid email format")

        self.user_info = {"name": "surajsapkota", "phone": 9864423841, "email": "surajsapkota50@gmail.com"}
        return self.user_info

    def book_appointment(self, date_str: str, department: str = "General Medicine") -> str:
        """Books an appointment with date validation."""
        date = self.parse_date(date_str)
        if not date:
            return "Invalid date format. Please use YYYY-MM-DD or specify a natural date like 'next Monday'."

        if datetime.strptime(date, '%Y-%m-%d') < datetime.now():
            return "Cannot book appointments for past dates."

        appointment = {"date": date, "department": department, "user": self.user_info}
        self.appointments.append(appointment)

        return f"""
✅ Appointment confirmed at {self.hospital_name}!
📅 Date: {appointment['date']}
🏥 Department: {department}
👤 Patient: {self.user_info.get('name', 'Unknown')}
📞 Contact: {self.user_info.get('phone', 'Unknown')}
✉️ Email: {self.user_info.get('email', 'Unknown')}
"""

    def setup_agent(self):
        """Initializes the agent with hospital tools."""
        tools = [
            Tool(name="Hospital Information", func=self.qa_chain.invoke,
                 description="Provides hospital-related information."),
            Tool(name="Book Appointment", func=self.book_appointment,
                 description="Books an appointment at the hospital.")
        ]

        self.agent = initialize_agent(
            tools=tools,
            llm=llm,
            agent=AgentType.OPENAI_FUNCTIONS,  # More structured function calling
            verbose=True
        )

    def chat(self, query: str) -> str:
        """Processes user queries using the agent."""
        return self.agent.invoke(query)


def main():
    chatbot = HospitalChatbot()

    print("🤖 Welcome to Surya Hospital Virtual Assistant!")
    print("Type 'quit' to exit.")

    while True:
        query = input("\nYou: ").strip()
        if query.lower() == 'quit':
            print("👋 Goodbye!")
            break

        if not query:
            print("\nBot: Please ask a question.")
            continue  # Skip processing if query is empty

        response = chatbot.chat(query)
        print("\nBot:", response)



if __name__ == "__main__":
    main()
