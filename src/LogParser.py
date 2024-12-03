import json
from pathlib import Path
from typing import List, Dict, Optional


class LogParser:
    """
    A class to parse log files and extract information related to patients, sessions, and protocols.
    """

    def __init__(self, log_path: str):
        """
        Initialize the LogParser with the path to the logs directory.

        Args:
            log_path (str): The path to the directory containing log files.
        """
        self.log_path = Path(log_path)

    def parse_single_log(self, path: str):
        """
        Parse a single log file and return its content as a dictionary.

        Args:
            path (str): The path to the log file.

        Returns:
            dict or None: The parsed log data, or None if parsing fails.
        """
        with open(path) as f:
            try:
                log = json.loads(json.load(f))
            except:
                return None
        return log

    def get_patient_ids(self) -> List[str]:
        """
        Retrieve a list of patient IDs.

        Each patient has a dedicated directory where the directory name is the patient ID.

        Returns:
            List[str]: A list of patient IDs.
        """
        # Each patient has a dedicated directory where dir_name = id
        patient_ids = [i.name for i in self.log_path.glob("*") if i.is_dir()]
        return patient_ids

    def get_session_paths(self, patient_id: str) -> List[str]:
        """
        Retrieve a list of session file paths for a given patient.

        Each session is represented by a JSON file.

        Args:
            patient_id (str): The ID of the patient.

        Returns:
            List[str]: A list of paths to session JSON files.
        """
        patient_dir = Path(self.log_path, patient_id)
        # Each session has a dedicated JSON file
        sessions = [i for i in patient_dir.glob("*.json")]
        # TODO: Return list of session objects
        return sessions

    def get_played_protocols(self, patient_id: str) -> Dict[str, int]:
        """
        Return a dictionary with protocol names as keys and number of sessions as values.

        Args:
            patient_id (str): The ID of the patient.

        Returns:
            Dict[str, int]: A dictionary mapping protocol names to the number of sessions played.
        """
        played_protocols = {}
        sessions = self.get_session_paths(patient_id=patient_id)
        for session in sessions:
            log = self.parse_single_log(session)
            protocol = log['Header']['ProtocolInfo']['ProtocolName']
            if protocol not in played_protocols:
                played_protocols[protocol] = 0
            played_protocols[protocol] += 1
        return played_protocols

    def get_dms(self, patient_id: str, protocol: Optional[str] = None):
        """
        Return a dictionary with difficulty modulators (DMs) for a given protocol.

        Returns DMs for all protocols if protocol name is not provided.

        Args:
            patient_id (str): The ID of the patient.
            protocol (str, optional): The name of the protocol. Defaults to None.

        Returns:
            dict: A nested dictionary containing DMs for the specified protocol(s).
        """
        sessions = self.get_session_paths(patient_id=patient_id)
        dms = {}
        for session in sessions:
            log = self.parse_single_log(session)
            if not log:
                continue
            session_protocol = log['Header']['ProtocolInfo']['ProtocolName']
            if protocol and session_protocol != protocol:
                continue
            if session_protocol not in dms:
                dms[session_protocol] = {}

            dm_logs = log['DifficultyParameters']['DifficultyModulators']  # List of dicts
            for item in dm_logs:
                game_mode = item['CurrentGameMode']
                if game_mode not in dms[session_protocol]:
                    dms[session_protocol][game_mode] = {}
                if item['key'] not in dms[session_protocol][game_mode]:
                    dms[session_protocol][game_mode][item['key']] = []
                dms[session_protocol][game_mode][item['key']].append(item['value'])
        return dms

    def get_hits_errors(self, patient_id: str, protocol: Optional[str] = None):
        """
        Return a dictionary with hits and errors for a given protocol.

        Returns data for all protocols if protocol name is not provided.

        Args:
            patient_id (str): The ID of the patient.
            protocol (str, optional): The name of the protocol. Defaults to None.

        Returns:
            dict: A nested dictionary containing hits and errors for the specified protocol(s).
        """
        sessions = self.get_session_paths(patient_id=patient_id)
        hits_and_errors = {}
        for session in sessions:
            log = self.parse_single_log(session)
            if not log:
                continue
            session_protocol = log['Header']['ProtocolInfo']['ProtocolName']
            if protocol and session_protocol != protocol:
                continue
            if session_protocol not in hits_and_errors:
                hits_and_errors[session_protocol] = {}

            playing_events = log['ProtocolEvents']['PlayingEvents']  # List of dicts
            for event in playing_events:
                game_mode = event['CurrentGameMode']
                if game_mode not in hits_and_errors[session_protocol]:
                    hits_and_errors[session_protocol][game_mode] = []
                value = 1 if event['Event'] == 'HIT' else 0
                hits_and_errors[session_protocol][game_mode].append(value)
        return hits_and_errors


