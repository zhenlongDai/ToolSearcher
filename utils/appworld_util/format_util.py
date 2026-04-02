import re

categories = [
  'api_docs',
  'supervisor',
  'amazon',
  'phone',
  'file_system',
  'spotify',
  'venmo',
  'gmail',
  'simple_note',
  'todoist',
]

app_description = {
  'api_docs': 'An app to search and explore API documentation.', 
  'supervisor': "An app to access supervisor's personal information, account credentials, addresses, payment cards, and manage the assigned task.", 
  'amazon': 'An online shopping app to buy products and manage orders, returns, etc.', 
  'phone': 'An app to find and manage contact information for friends, family members, etc., send and receive messages, and manage alarms.', 
  'file_system': 'A file system app to create and manage files and folders.', 
  'spotify': 'A music streaming app to stream songs and manage song, album and playlist libraries.', 
  'venmo': 'A social payment app to send, receive and request money to and from others.', 
  'gmail': 'An email app to draft, send, receive, and manage emails.', 
  'splitwise': 'A bill splitting app to track and split expenses with people.', 
  'simple_note': 'A note-taking app to create and manage notes.', 
  'todoist': 'A task management app to manage todo lists and collaborate on them with others.'}