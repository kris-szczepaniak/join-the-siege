from src.app import app

if __name__ == "__main__":
    """
    I work on a mac.
    Ports 5000 and 7000 are used for Control Centre starting from Mac OSX 12.x
    """
    app.run(port=5001, debug=False)
