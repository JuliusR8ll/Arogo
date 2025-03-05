import numpy as np
import joblib
from flask import Flask, request, render_template, send_file, jsonify, redirect, url_for, session
from flask_sqlalchemy import SQLAlchemy
from flask_login import LoginManager, UserMixin, login_user, login_required, logout_user, current_user
import io
from utils import generate_pdf, call_gemini_api
import bcrypt
from flask_socketio import SocketIO, emit, join_room, leave_room

app = Flask(__name__)
app.secret_key = 'your_secret_key'  # Change this to a random secret key
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///users.db'  # Use SQLite for simplicity
db = SQLAlchemy(app)
login_manager = LoginManager()
login_manager.init_app(app)

# Initialize SocketIO
socketio = SocketIO(app)

# User model
class User(UserMixin, db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(150), unique=True, nullable=False)
    password = db.Column(db.LargeBinary, nullable=False)  # Store as LargeBinary for hashed password
    role = db.Column(db.String(50), nullable=False)  # 'tester' or 'psychiatrist'

class Message(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    sender_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    receiver_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    content = db.Column(db.Text, nullable=False)
    timestamp = db.Column(db.DateTime, default=db.func.current_timestamp())

class Psychiatrist(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(150), unique=True, nullable=False)
    bio = db.Column(db.Text, nullable=True)  # Optional bio for the psychiatrist

@login_manager.user_loader
def load_user(user_id):
    return User.query.get(int(user_id))

@app.route('/')
def home_redirect():
    return redirect(url_for('login'))  # Always redirect to login page

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        
        user = User.query.filter_by(username=username).first()
        if user and bcrypt.checkpw(password.encode('utf-8'), user.password):
            login_user(user)
            return redirect(url_for('home'))  # Redirect to the home page after login
        else:
            return "Invalid username or password. Please try again."

    return render_template('login.html')  # Render the login page

@app.route('/logout')
@login_required
def logout():
    logout_user()
    return redirect(url_for('login'))  # Redirect to login after logout

@socketio.on('join')
def on_join(data):
    room = f"chat_{data['psychiatrist_id']}"
    join_room(room)
    emit('status', {'msg': f"{current_user.username} has joined the room."}, room=room)

@socketio.on('message')
def handle_message(data):
    room = f"chat_{data['psychiatrist_id']}"
    message = data['message']
    
    # Save message to database
    new_message = Message(
        sender_id=current_user.id,
        receiver_id=data['psychiatrist_id'],
        content=message
    )
    db.session.add(new_message)
    db.session.commit()
    
    # Emit message to room
    emit('message', {
        'sender': current_user.username,
        'message': message,
        'timestamp': new_message.timestamp.strftime('%Y-%m-%d %H:%M:%S')
    }, room=room)

@app.route('/chat/<int:psychiatrist_id>')
@login_required
def chat(psychiatrist_id):
    messages = Message.query.filter(
        (Message.sender_id == current_user.id) | (Message.receiver_id == current_user.id)
    ).filter(
        (Message.sender_id == psychiatrist_id) | (Message.receiver_id == psychiatrist_id)
    ).all()
    
    return render_template('chat.html', 
                         user=current_user, 
                         messages=messages, 
                         psychiatrist_id=psychiatrist_id)

@app.route('/home')
@login_required  # Ensure that only logged-in users can access this route
def home():
    return render_template('index.html', user=current_user)  # Pass the current user to the template

@app.route('/predict', methods=['POST'])
def predict():
    depression_model = joblib.load('depression_model.pkl')

    age = float(request.form['age'])
    
    gender = request.form['gender']
    gender_encoded = 1 if gender == 'male' else 0 
    bmi = float(request.form['bmi'])
    phq_score = float(request.form['phq_score'])
    
    anxiety_severity = float(request.form['anxiety_severity'])  
    
    epworth_score = float(request.form['epworth_score'])

    gad_score = float(request.form['gad_score'])  

    input_data = np.array([[age, gender_encoded, bmi, phq_score, anxiety_severity, epworth_score, gad_score]], dtype=float)
    
    prediction = depression_model.predict(input_data)
    
    api_input = {
        "age": age, 
        "gender": gender_encoded,
        "bmi": bmi,
        "phq_score": phq_score,
        "anxiety_severity": anxiety_severity,
        "epworth_score": epworth_score,
        "gad_score": gad_score,
        "predicted_severity": prediction.tolist() 
    }

    # Call Gemini API to get suggestions
    suggestions = call_gemini_api(api_input)
    if not suggestions:
        return "No suggestions returned from the API."

    # Generate PDF
    pdf = generate_pdf(suggestions)
    if pdf is None:
        return "Failed to generate PDF."

    pdf_io = io.BytesIO(pdf)
    pdf_io.seek(0)

    # Send the PDF as a response
    return send_file(pdf_io, as_attachment=True, download_name='suggestions.pdf', mimetype='application/pdf')

@app.route('/calculate_gad', methods=['POST'])
def calculate_gad():
    gad_questions = [
        "Feeling nervous, anxious, or on edge",
        "Not being able to stop or control worrying",
        "Worrying too much about different things",
        "Trouble relaxing",
        "Being so restless that it's hard to sit still",
        "Becoming easily annoyed or irritable",
        "Feeling afraid as if something awful might happen"
    ]

    gad_scores = 0
    for question in gad_questions:
        response = request.form.get(question)  
        if response == "Not at all":
            gad_scores += 0
        elif response == "Several days":
            gad_scores += 1
        elif response == "More than half the days":
            gad_scores += 2
        elif response == "Nearly every day":
            gad_scores += 3

    return f"Your GAD score is: {gad_scores}"

@app.route('/gad_score')
def gad_score():
    return render_template('gad_score.html')

@app.route('/calculate_phq', methods=['POST'])
def calculate_phq():
    phq_questions = [
        "phq_1", 
        "phq_2", 
        "phq_3", 
        "phq_4", 
        "phq_5",  
        "phq_6",  
        "phq_7", 
        "phq_8", 
        "phq_9"  
    ]

    phq_score = 0
    for question in phq_questions:
        response = request.form.get(question)
        print(request.form)
        if response == "Not at all":
            phq_score += 0
        elif response == "Several days":
            phq_score += 1
        elif response == "More than half the days":
            phq_score += 2
        elif response == "Nearly every day":
            phq_score += 3

    return f"Your PHQ score is: {phq_score}"

@app.route('/phq_score')
def phq_score():
    return render_template('phq_score.html')

@app.route('/calculate_epworth', methods=['POST'])
def calculate_epworth():
    epworth_questions = [
        "epworth_1",
        "epworth_2",
        "epworth_3",
        "epworth_4",
        "epworth_5",
        "epworth_6",
        "epworth_7",
        "epworth_8"
    ]

    epworth_score = 0
    for question in epworth_questions:
        response = int(request.form.get(question, 0))  
        epworth_score += response

    return f"Your Epworth score is: {epworth_score}"

@app.route('/epworth_score')
def epworth_score():
    return render_template('epworth_score.html')

@app.route('/calculate_bmi', methods=['POST'])
def calculate_bmi():
    weight = float(request.form['weight'])
    height = float(request.form['height'])
    
    # Calculate BMI
    bmi = weight / (height ** 2)

    return f"Your BMI is: {bmi:.2f}"

@app.route('/bmi_calculator')
def bmi_calculator():
    return render_template('bmi_calculator.html')

@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        role = request.form['role']  # 'tester' or 'psychiatrist'

        # Check if the username already exists
        if User.query.filter_by(username=username).first():
            return "Username already exists. Please choose a different one."

        # Hash the password
        hashed_password = bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt())

        # Create a new user
        new_user = User(username=username, password=hashed_password, role=role)
        db.session.add(new_user)
        db.session.commit()

        # If the user is a psychiatrist, create a psychiatrist profile
        if role == 'psychiatrist':
            new_psychiatrist = Psychiatrist(username=username)
            db.session.add(new_psychiatrist)
            db.session.commit()

        return redirect(url_for('login'))  # Redirect to login after registration

    return render_template('register.html')

@app.route('/psychiatrists')
@login_required
def psychiatrists():
    all_psychiatrists = Psychiatrist.query.all()
    return render_template('psychiatrists.html', psychiatrists=all_psychiatrists)

@app.route('/chat/<int:psychiatrist_id>', methods=['GET', 'POST'])
@login_required
def chat_with_psychiatrist(psychiatrist_id):
    if request.method == 'POST':
        message = request.form['message']
        # Save message to the database
        new_message = Message(sender_id=current_user.id, receiver_id=psychiatrist_id, content=message)
        db.session.add(new_message)
        db.session.commit()
        return redirect(url_for('chat_with_psychiatrist', psychiatrist_id=psychiatrist_id))

    # Retrieve messages for the chat
    messages = Message.query.filter(
        (Message.sender_id == current_user.id) | (Message.receiver_id == current_user.id)
    ).filter(
        (Message.sender_id == psychiatrist_id) | (Message.receiver_id == psychiatrist_id)
    ).all()

    return render_template('chat.html', user=current_user, messages=messages, psychiatrist_id=psychiatrist_id)

if __name__ == '__main__':
    db.create_all()  # Create database tables
    socketio.run(app, debug=True)
