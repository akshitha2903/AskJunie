import logging
from langchain.prompts.chat import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
)

class Validation:
    @staticmethod
    def parse_output(output: str):
        """
        Custom parser to validate and structure the output.
        """
        try:
            lines = output.splitlines()
            result = {}
            for line in lines:
                if "=" in line:
                    key, value = line.split("=", 1)
                    result[key.strip()] = value.strip()
            
            # If no explicit validation, default to yes
            if 'plan_is_valid' not in result:
                result['plan_is_valid'] = 'yes'
            
            return result
        except Exception as e:
            logging.error(f"Error parsing output: {e}")
            return {"plan_is_valid": "yes", "updated_request": "N/A"}

class ValidationTemplate:
    def __init__(self):
        self.system_template = """
        You are a travel expert who helps users with various travel-related queries.

        The user's request will be denoted by four hashtags. Your job is to determine 
        if the request is meaningful and processable.

        Validation Criteria:
        - Travel tips and general inquiries are ALWAYS valid
        - Requests for destination advice are valid
        - Specific travel planning queries are valid
        - Any request related to travel in any form should be considered valid

        Guidelines:
        - If the query is travel-related in ANY way, it is valid
        - For broad or general queries, consider them valid
        - Only mark as invalid if the request is completely unrelated to travel
        - If unsure, default to marking as valid

        Output Format:
        plan_is_valid = [yes/no]
        updated_request = [Your clarification or "N/A"]
        """

        self.human_template = "####{query}####"
        
        self.system_message_prompt = SystemMessagePromptTemplate.from_template(self.system_template)
        self.human_message_prompt = HumanMessagePromptTemplate.from_template(
            self.human_template, input_variables=["query"]
        )
        self.chat_prompt = ChatPromptTemplate.from_messages(
            [self.system_message_prompt, self.human_message_prompt]
        )