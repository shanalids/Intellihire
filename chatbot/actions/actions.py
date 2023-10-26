from typing import Any, Text, Dict, List
from rasa_sdk import Action, Tracker,FormValidationAction
from rasa_sdk.events import SlotSet
from rasa_sdk.executor import CollectingDispatcher
import csv
from rasa_sdk.types import DomainDict
import requests
from flask import session
import csv

from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph
from reportlab.lib.styles import getSampleStyleSheet

class ValidateNameForm(FormValidationAction):
    def name(self) -> Text:
        return "validate_name_form"

    def validate_first_name(
        self,
        slot_value: Any,
        dispatcher: CollectingDispatcher,
        tracker: Tracker,
        domain: DomainDict,
    ) -> Dict[Text, Any]:
        """Validate `first_name` value."""

        # If the user is typing a super short name, it might be misspelled.
        # Note that this is an assumption and it might not hold true depending
        # on where your users live.

        if len(slot_value) <= 2:
            dispatcher.utter_message(text=f"That's a very short name. I'm assuming you mis-spelled.")
            return {"first_name": None}
        else:
            return {"first_name": slot_value}

    def validate_last_name(
        self,
        slot_value: Any,
        dispatcher: CollectingDispatcher,
        tracker: Tracker,
        domain: DomainDict,
    ) -> Dict[Text, Any]:
        """Validate `last_name` value."""

        # If the user is typing a super short name, it might be misspelled.
        # Note that this is an assumption and it might not hold true depending
        # on where your users live.

        if len(slot_value) <= 2:
            dispatcher.utter_message(text=f"That's a very short name. I'm assuming you mis-spelled.")
            return {"last_name": None}
        else:
            return {"last_name": slot_value}

    def validate_job_title(
        self,
        slot_value: Any,
        dispatcher: CollectingDispatcher,
        tracker: Tracker,
        domain: DomainDict,
    ) -> Dict[Text, Any]:
        """Validate `job_title` value."""



        if len(slot_value) <= 2:
            dispatcher.utter_message(text=f"That's a very short job_title. I'm assuming you mis-spelled.")
            return {"job_title": None}
        else:
            return {"job_title": slot_value}

    def validate_job_summery(
            self,
            slot_value: Any,
            dispatcher: CollectingDispatcher,
            tracker: Tracker,
            domain: DomainDict,
    ) -> Dict[Text, Any]:
        """Validate `job_summery` value."""


        if len(slot_value) <= 2:
            dispatcher.utter_message(text=f"That's a very short job_summery. I'm assuming you mis-spelled.")
            return {"job_summery": None}
        else:
            return {"job_summery": slot_value}

    def validate_job_Responsibilities(
            self,
            slot_value: Any,
            dispatcher: CollectingDispatcher,
            tracker: Tracker,
            domain: DomainDict,
    ) -> Dict[Text, Any]:
        """Validate `job_Responsibilities` value."""


        if len(slot_value) <= 2:
            dispatcher.utter_message(text=f"That's a very short job_Responsibilities. I'm assuming you missed.")
            return {"job_Responsibilities": None}
        else:
            return {"job_Responsibilities": slot_value}

    def validate_job_Qualifications(
            self,
            slot_value: Any,
            dispatcher: CollectingDispatcher,
            tracker: Tracker,
            domain: DomainDict,
    ) -> Dict[Text, Any]:
        """Validate `job_Qualifications` value."""


        if len(slot_value) <= 2:
            dispatcher.utter_message(text=f"That's a very short job_Qualifications. I'm assuming you missed.")
            return {"job_Qualifications": None}
        else:
            return {"job_Qualifications": slot_value}

    def validate_job_Education(
            self,
            slot_value: Any,
            dispatcher: CollectingDispatcher,
            tracker: Tracker,
            domain: DomainDict,
    ) -> Dict[Text, Any]:
        """Validate `job_Education` value."""


        if len(slot_value) <= 2:
            dispatcher.utter_message(text=f"That's a very short job_Education. I'm assuming you missed.")
            return {"job_Education": None}
        else:
            return {"job_Education": slot_value}


    def validate_job_Employment_Type(
            self,
            slot_value: Any,
            dispatcher: CollectingDispatcher,
            tracker: Tracker,
            domain: DomainDict,
    ) -> Dict[Text, Any]:
        """Validate `job_Employment_Type` value."""


        if len(slot_value) <= 2:
            dispatcher.utter_message(text=f"That's a very short job_Employment_Type. I'm assuming you missed.")
            return {"job_Employment_Type": None}
        else:
            return {"job_Employment_Type": slot_value}


    def validate_job_Work_Schedule(
            self,
            slot_value: Any,
            dispatcher: CollectingDispatcher,
            tracker: Tracker,
            domain: DomainDict,
    ) -> Dict[Text, Any]:
        """Validate `job_Work_Schedule` value."""


        if len(slot_value) <= 2:
            dispatcher.utter_message(text=f"That's a very short job_Work_Schedule. I'm assuming you missed.")
            return {"job_Work_Schedule": None}
        else:
            return {"job_Work_Schedule": slot_value}

    def validate_job_Location(
            self,
            slot_value: Any,
            dispatcher: CollectingDispatcher,
            tracker: Tracker,
            domain: DomainDict,
    ) -> Dict[Text, Any]:
        """Validate `job_Location` value."""


        if len(slot_value) <= 2:
            dispatcher.utter_message(text=f"That's a very short job_Location. I'm assuming you missed.")
            return {"job_Location": None}
        else:
            return {"job_Location": slot_value}




    def validate_job_Salary(
            self,
            slot_value: Any,
            dispatcher: CollectingDispatcher,
            tracker: Tracker,
            domain: DomainDict,
    ) -> Dict[Text, Any]:
        """Validate `job_Salary` value."""


        if len(slot_value) <= 2:
            dispatcher.utter_message(text=f"That's a very short job_Salary. I'm assuming you missed.")
            return {"job_Salary": None}
        else:
            return {"job_Salary": slot_value}


    def validate_job_Application_Process(
            self,
            slot_value: Any,
            dispatcher: CollectingDispatcher,
            tracker: Tracker,
            domain: DomainDict,
    ) -> Dict[Text, Any]:
        """Validate `job_Application_Process` value."""


        if len(slot_value) <= 2:
            dispatcher.utter_message(text=f"That's a very short job_Application_Process. I'm assuming you missed.")
            return {"job_Application_Process": None}
        else:
            return {"job_Application_Process": slot_value}


class PrintSlotsAction(Action):
    def name(self) -> Text:
        return "action_print_slots"

    def run(self, dispatcher: CollectingDispatcher,
            tracker: Tracker,
            domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:

        

        first_name = tracker.get_slot("first_name")
        last_name = tracker.get_slot("last_name")
        job_title = tracker.get_slot("job_title")
        job_summery = tracker.get_slot("job_summery")
        job_Responsibilities = tracker.get_slot("job_Responsibilities")
        job_Qualifications = tracker.get_slot("job_Qualifications")
        job_Education = tracker.get_slot("job_Education")
        job_Employment_Type = tracker.get_slot("job_Employment_Type")
        job_Work_Schedule = tracker.get_slot("job_Work_Schedule")
        job_Location = tracker.get_slot("job_Location")
        job_Salary = tracker.get_slot("job_Salary")
        job_Application_Process = tracker.get_slot("job_Application_Process")



        slot_message = (
            f"Created by: {first_name} {last_name}\n"
            f"Job Title: {job_title}\n"
            f"Job Summary: {job_summery}\n"
            f"Job Responsibilities: {job_Responsibilities}\n"
            f"Job Qualifications: {job_Qualifications}\n"
            f"Job Education: {job_Education}\n"
            f"Job Employment Type: {job_Employment_Type}\n"
            f"Job Work Schedule: {job_Work_Schedule}\n"
            f"Job Location: {job_Location}\n"
            f"Job Salary: {job_Salary}\n"
            f"Job Application Process: {job_Application_Process}\n"
           
        )

            # Create a list of slot values
        slot_values = [
            first_name,
            last_name,
            job_title,
            job_summery,
            job_Responsibilities,
            job_Qualifications,
            job_Education,
            job_Employment_Type,
            job_Work_Schedule,
            job_Location,
            job_Salary,
            job_Application_Process,
        ]

        pdf_filename = "output.pdf"
        doc = SimpleDocTemplate(pdf_filename, pagesize=letter)
        styles = getSampleStyleSheet()
        story = []
        for slot_name, slot_value in zip(tracker.slots.keys(), slot_values):
            story.append(Paragraph(f"{slot_name}: {slot_value}", styles['Normal']))
        doc.build(story)

        dispatcher.utter_message(text=f"I've created a PDF with the slot values. ")
        
        #dispatcher.utter_message(text=slot_values)


        
        
        return []

class ActionRunInteractiveStory1(Action):
    def name(self) -> Text:
        return "action_run_interactive_story_1"

    def run(
        self,
        dispatcher: CollectingDispatcher,
        tracker: Tracker,
        domain: Dict[Text, Any],
    ) -> List[Dict[Text, Any]]:
        # Implement the logic to restart or refresh the interactive story here
        dispatcher.utter_message("Restarting the interactive story...")
        return []