import datetime
import subprocess
import os
import shutil
results_dir = "./results"
keyword = ": "

def get_current_git_commit():
    try:
        commit = subprocess.check_output(["git", "rev-parse", "HEAD"]).strip()
        commit_message = subprocess.check_output(["git", "log", "-1", "--pretty=%B"]).strip()
        return commit.decode('utf-8'), commit_message.decode('utf-8')
    except Exception as e:
        print(f"Error: {str(e)}")
        return None, None

def copy_logs(src_dir, dst_dir, base_name='logs'):
    """
    Copies 'logs' directory from source to destination. 
    If 'logs' already exists at the destination, appends a suffix to create a new directory.
    """

    src_logs = os.path.join(src_dir, base_name)

    if not os.path.exists(src_logs):
        print(f"Source logs directory {src_logs} does not exist.")
        return

    counter = 0
    while True:
        suffix = f"_{counter}" if counter else ""
        dst_logs = os.path.join(dst_dir, base_name + suffix)
        try:
            shutil.copytree(src_logs, dst_logs)
            break
        except FileExistsError:
            counter += 1
            
    return dst_logs

# Ask for a user message
user_msg = input("Enter a message for the user: ")

# Get the current git commit and commit message
commit, commit_message = get_current_git_commit()

assert keyword in commit_message, f"Commit message must contain with {keyword}"
msg = commit_message.split(keyword, 1)[-1]

# Get the current date in format YYYY_MM_DD
date = datetime.datetime.now().strftime("%Y_%m_%d")


# Check if the results/date/logs directory exists
dst_logs = copy_logs(".", f"{results_dir}/{date}")

# Create a notes.txt file in the results/date directory
# and write the commit and commit message to it
with open(f"{dst_logs}/notes.txt", "w") as f:
    if user_msg:
        f.write(f"User message: {user_msg}\n")
    f.write(f"Commit message: {msg}\n")
    f.write(f"Commit: {commit}\n")
    
    





