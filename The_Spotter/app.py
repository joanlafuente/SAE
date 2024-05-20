

import os
import sqlite3
from flask import Flask, render_template, request, redirect, url_for, flash, session, send_from_directory
from flask_session import Session
from werkzeug.security import check_password_hash, generate_password_hash
from werkzeug.utils import secure_filename
from pyvis.network import Network


# Configure application
app = Flask(__name__)

# Configure session to use filesystem (instead of signed cookies)
app.config["SESSION_PERMANENT"] = False
app.config["SESSION_TYPE"] = "filesystem"
Session(app)

@app.after_request
def after_request(response):
    """Ensure responses aren't cached"""
    response.headers["Cache-Control"] = "no-cache, no-store, must-revalidate"
    response.headers["Expires"] = 0
    response.headers["Pragma"] = "no-cache"
    return response

# Function to get database connection
def get_db_connection(db_name):
    conn = sqlite3.connect(db_name)
    conn.row_factory = sqlite3.Row  # to access columns by name
    return conn

@app.route("/")
def index():
    """Show index page"""

    return render_template("index.html")

@app.route("/index.html")
def index_():
    """Show index page"""

    return render_template("index.html")

@app.route('/about_us.html')
def about_us():
    return render_template('about_us.html')

@app.route('/create_acc.html')
def create_acc():
    return render_template('create_acc.html')

@app.route('/log_in.html')
def log_in():
    return render_template('log_in.html')

@app.route('/playground.html')
def playground():
    return render_template('playground.html')

@app.route('/playground_graph1')
def playground_graph1_route():
    # Generate the second graph
    graph_file, graph= generate_playground_graph1()
    print(graph_file)
    # Return the file path of the second graph
    return render_template('graph.html', graph_file=graph_file)

def generate_playground_graph1():
    # Create a pyvis network object
    graph = Network(select_menu=True, cdn_resources='remote', filter_menu=True)

    # Add nodes and edges to the graph (example)
    graph.add_node(1, label='Node 1')
    graph.add_node(2, label='Node 2')
    graph.add_edge(1, 2)

    graph.show_buttons(filter_=['nodes', 'edges', 'physics'])
    # Save the graph to a file
    graph_file = os.path.join(app.static_folder, 'graph1.html')
    graph.write_html(graph_file)
    print("graph_file1")
    # Render the playground page with the graph
    return '/static/graph1.html', graph


@app.route('/playground_graph2')
def playground_graph2_route():
    # Generate the second graph
    graph_file, graph = generate_playground_graph2()
    print(graph_file)
    # Return the file path of the second graph
    return render_template('graph.html', graph_file=graph_file)

def generate_playground_graph2():
    # Create a pyvis network object
    graph = Network(cdn_resources='remote', filter_menu=True, directed=False, select_menu=True)

    # Add nodes and edges to the graph (example)
    graph.add_node(1, label='Node 1', color='red')
    graph.add_node(2, label='Node 2')
    graph.add_node(3, label='Node 3')
    graph.add_node(4, label='Node 4')
    graph.add_edge(1, 2)
    graph.add_edge(1, 3)
    graph.add_edge(3, 2)
    graph.add_edge(4, 2)

    graph.show_buttons(filter_=['physics'])
    # Save the graph to a file
    graph_file = os.path.join(app.static_folder, 'graph2.html')
    graph.write_html(graph_file)
    print("graph_file2")
    # Return the file path and the graph object
    return '/static/graph2.html', graph



# Add route to serve the graph file
@app.route('/static/<path:path>')
def send_static(path):
    return send_from_directory('static', path)


@app.route("/login", methods=["GET", "POST"])
def login():
    """Log user in"""

    # Forget any user_id
    session.clear()

    # User reached route via POST (as by submitting a form via POST)
    if request.method == "POST":
        # Ensure username was submitted

        username = request.form.get("username")
        password = request.form.get("password")

        # Enusre username and password were submitted
        if not username or not password:
            flash("Must provide username and password", "warning")
            return render_template("login.html")
        
        # Query database for username
        conn = get_db_connection("users.db")
        user = conn.execute("SELECT * FROM users WHERE username = ?", (request.form.get("username"),)).fetchone()
        conn.close()

        # Ensure username exists and password is correct
        if user is None or not check_password_hash(user["password"], request.form.get("password")):
            flash("Invalid username and/or password")
            return render_template("login.html")
        
        # Remember which user has logged in
        session["user_id"] = user["id"]
        session["username"] = user["username"]

        # Redirect user to home page
        flash("You have successfully logged in")
        return redirect("/index.html")
    else:
        return render_template("login.html")

@app.route("/register", methods=["GET", "POST"])
def register():
    """Register user"""

    # User reached route via POST (as by submitting a form via POST)
    if request.method == "POST":

        # Ensure username and password were submitted
        if not request.form.get("username") or not request.form.get("password"):
            flash("Must provide username and password", "warning")
            return render_template("register.html")
        
        # Ensure password and confirmation match
        if request.form.get("password") != request.form.get("confirmation"):
            flash("Password and confirmation must match", "warning")
            return render_template("register.html")
        
        # Query database for username
        conn = get_db_connection("users.db")
        username = conn.execute("SELECT * FROM users WHERE username = ?", (request.form.get("username"),)).fetchone()
        conn.close()

        # Ensure username does not already exist
        if username:
            flash("Username already exists", "warning")
            return render_template("register.html")
        
        # Insert new user into database
        conn = get_db_connection("users.db")
        conn.execute("INSERT INTO users (username, password) VALUES (?, ?)", (request.form.get("username"), generate_password_hash(request.form.get("password"))))
        conn.commit()
        conn.close()

        # Redirect user to login page
        flash("You have successfully registered")
        return redirect("/login")
    else:
        return render_template("register.html")

if __name__ == "__main__":
    app.run(debug=True, port=8082)


