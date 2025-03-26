from flask import jsonify, make_response
from functools import wraps


def error_interceptor(f):
    """
    Intercepts unhandled errors from function calls inside the endpoint.
    """
    @wraps(f)
    def wrapper(*args, **kwargs):
        try:
            return f(*args, **kwargs)
        except Exception as e:
            print(f"Error intercepted: {e}")
            return make_response(jsonify({"error": str(e)})), 400
        
    return wrapper