#!/usr/bin/env python
import sys
import warnings

from datetime import datetime

from hawk.crew import Hawk
from crewai_tools import SerperDevTool

warnings.filterwarnings("ignore", category=SyntaxWarning, module="pysbd")

# This main file is intended to be a way for you to run your
# crew locally, so refrain from adding unnecessary logic into this file.
# Replace with inputs you want to test with, it will automatically
# interpolate any tasks and agents information


def run():
    """
    Run the crew.
    """
    inputs = {
        'topic': 'ClawBot',
        'current_year': str(datetime.now().year)
    }
    result = Hawk().crew().kickoff(inputs=inputs)
    print(result.raw)
if __name__ == "__main__":
    run()
