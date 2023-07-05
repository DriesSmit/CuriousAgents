import subprocess

def get_current_git_commit():
    try:
        commit = subprocess.check_output(["git", "rev-parse", "HEAD"]).strip()
        return commit.decode('utf-8')
    except Exception as e:
        print(f"Error: {str(e)}")
        return None
    
print(get_current_git_commit())