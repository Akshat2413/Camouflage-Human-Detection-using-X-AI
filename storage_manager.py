"""
Storage Manager - Handles saving and loading scan sessions
"""

import json
import shutil
from pathlib import Path
from datetime import datetime


class StorageManager:
    """Manages storage of scan sessions and chat history"""
    
    def __init__(self, base_dir="app_data"):
        self.base_dir = Path(base_dir)
        self.sessions_dir = self.base_dir / "sessions"
        self.index_file = self.base_dir / "sessions_index.json"
        
        # Create directories
        self.sessions_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize index if doesn't exist
        if not self.index_file.exists():
            self._save_index([])
    
    def _save_index(self, sessions_list):
        """Save sessions index"""
        with open(self.index_file, 'w') as f:
            json.dump(sessions_list, f, indent=2)
    
    def _load_index(self):
        """Load sessions index"""
        if self.index_file.exists():
            with open(self.index_file, 'r') as f:
                return json.load(f)
        return []
    
    def create_session(self, name=None):
        """Create a new session"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        session_id = f"session_{timestamp}"
        
        if name is None:
            name = f"Scan {datetime.now().strftime('%Y-%m-%d %H:%M')}"
        
        # Create session folder
        session_path = self.sessions_dir / session_id
        session_path.mkdir(exist_ok=True)
        
        # Create subfolders
        (session_path / "images").mkdir(exist_ok=True)
        (session_path / "xai").mkdir(exist_ok=True)
        
        # Session metadata
        session_data = {
            "session_id": session_id,
            "name": name,
            "created": datetime.now().isoformat(),
            "last_modified": datetime.now().isoformat(),
            "scan_count": 0,
            "message_count": 0
        }
        
        # Save metadata
        with open(session_path / "metadata.json", 'w') as f:
            json.dump(session_data, f, indent=2)
        
        # Update index
        sessions = self._load_index()
        sessions.insert(0, session_data)
        self._save_index(sessions)
        
        return session_id
    
    def save_scan_results(self, session_id, original_image, detected_image, 
                         detections, statistics, chat_history):
        """Save scan results to session"""
        session_path = self.sessions_dir / session_id
        
        if not session_path.exists():
            raise ValueError(f"Session {session_id} not found")
        
        # Save images
        import cv2
        cv2.imwrite(str(session_path / "images" / "original.jpg"), original_image)
        cv2.imwrite(str(session_path / "images" / "detected.jpg"), detected_image)
        
        # Save detection results
        results = {
            "timestamp": datetime.now().isoformat(),
            "detections": detections,
            "statistics": statistics
        }
        
        with open(session_path / "results.json", 'w') as f:
            json.dump(results, f, indent=2)
        
        # Save chat history
        with open(session_path / "chat_history.json", 'w') as f:
            json.dump(chat_history, f, indent=2)
        
        # Update metadata
        with open(session_path / "metadata.json", 'r') as f:
            metadata = json.load(f)
        
        metadata["last_modified"] = datetime.now().isoformat()
        metadata["scan_count"] += 1
        metadata["message_count"] = len(chat_history)
        
        with open(session_path / "metadata.json", 'w') as f:
            json.dump(metadata, f, indent=2)
        
        # Update index
        self._update_index_entry(session_id, metadata)
    
    def _update_index_entry(self, session_id, metadata):
        """Update a session in the index"""
        sessions = self._load_index()
        for i, session in enumerate(sessions):
            if session["session_id"] == session_id:
                sessions[i] = metadata
                break
        self._save_index(sessions)
    
    def load_session(self, session_id):
        """Load a session's data"""
        session_path = self.sessions_dir / session_id
        
        if not session_path.exists():
            return None
        
        # Load metadata
        with open(session_path / "metadata.json", 'r') as f:
            metadata = json.load(f)
        
        # Load results if exists
        results = None
        if (session_path / "results.json").exists():
            with open(session_path / "results.json", 'r') as f:
                results = json.load(f)
        
        # Load chat history
        chat_history = []
        if (session_path / "chat_history.json").exists():
            with open(session_path / "chat_history.json", 'r') as f:
                chat_history = json.load(f)
        
        # Load images if exist
        import cv2
        original_image = None
        detected_image = None
        
        original_path = session_path / "images" / "original.jpg"
        detected_path = session_path / "images" / "detected.jpg"
        
        if original_path.exists():
            original_image = cv2.imread(str(original_path))
        
        if detected_path.exists():
            detected_image = cv2.imread(str(detected_path))
        
        return {
            "metadata": metadata,
            "results": results,
            "chat_history": chat_history,
            "original_image": original_image,
            "detected_image": detected_image
        }
    
    def list_sessions(self):
        """List all sessions"""
        return self._load_index()
    
    def delete_session(self, session_id):
        """Delete a session"""
        session_path = self.sessions_dir / session_id
        
        if session_path.exists():
            shutil.rmtree(session_path)
        
        # Update index
        sessions = self._load_index()
        sessions = [s for s in sessions if s["session_id"] != session_id]
        self._save_index(sessions)
        
        return True
    
    def rename_session(self, session_id, new_name):
        """Rename a session"""
        session_path = self.sessions_dir / session_id
        
        if not session_path.exists():
            return False
        
        # Update metadata
        with open(session_path / "metadata.json", 'r') as f:
            metadata = json.load(f)
        
        metadata["name"] = new_name
        metadata["last_modified"] = datetime.now().isoformat()
        
        with open(session_path / "metadata.json", 'w') as f:
            json.dump(metadata, f, indent=2)
        
        # Update index
        self._update_index_entry(session_id, metadata)
        
        return True
    
    def get_statistics(self):
        """Get overall statistics"""
        sessions = self._load_index()
        
        total_scans = sum(s.get("scan_count", 0) for s in sessions)
        total_messages = sum(s.get("message_count", 0) for s in sessions)
        
        return {
            "total_sessions": len(sessions),
            "total_scans": total_scans,
            "total_messages": total_messages
        }
    
    def save_chat(self, session_id, user_message, assistant_response, message_type):
        """Save individual chat message - FIXED INDENTATION"""
        log_entry = {
            "session_id": session_id,
            "timestamp": datetime.now().isoformat(),
            "user": user_message,
            "assistant": assistant_response,
            "type": message_type
        }
        
        # Save to chat logs file
        chat_log_file = self.base_dir / "chat_logs.json"
        with open(chat_log_file, "a", encoding="utf-8") as f:
            f.write(json.dumps(log_entry) + "\n")