from flask import Flask, jsonify, request, render_template, redirect, url_for
from pymongo import MongoClient
from flask import jsonify, request, flash
from werkzeug.security import check_password_hash, generate_password_hash
import pickle
import numpy as np
from sklearn.preprocessing import OneHotEncoder
import json

from recommendation import filter_material

app = Flask(__name__,template_folder='templates',static_folder='static')
client=MongoClient('localhost', 27017)
# Configuration for MongoDB
db=client.flask_database
Users=db.Users
model = pickle.load(open('House_price.pickle', 'rb'))


all_locations=["1st block jayanagar", "1st phase jp nagar", "2nd phase judicial layout", "2nd stage nagarbhavi", "5th block hbr layout", "5th phase jp nagar", "6th phase jp nagar", "7th phase jp nagar", "8th phase jp nagar", "9th phase jp nagar", "aecs layout", "abbigere", "akshaya nagar", "ambalipura", "ambedkar nagar", "amruthahalli", "anandapura", "ananth nagar", "anekal", "anjanapura", "ardendale", "arekere", "attibele", "beml layout", "btm 2nd stage", "btm layout", "babusapalaya", "badavala nagar", "balagere", "banashankari", "banashankari stage ii", "banashankari stage iii", "banashankari stage v", "banashankari stage vi", "banaswadi", "banjara layout", "bannerghatta", "bannerghatta road", "basavangudi", "basaveshwara nagar", "battarahalli", "begur", "begur road", "bellandur", "benson town", "bharathi nagar", "bhoganhalli", "billekahalli", "binny pete", "bisuvanahalli", "bommanahalli", "bommasandra", "bommasandra industrial area", "bommenahalli", "brookefield", "budigere", "cv raman nagar", "chamrajpet", "chandapura", "channasandra", "chikka tirupathi", "chikkabanavar", "chikkalasandra", "choodasandra", "cooke town", "cox town", "cunningham road", "dasanapura", "dasarahalli", "devanahalli", "devarachikkanahalli", "dodda nekkundi", "doddaballapur", "doddakallasandra", "doddathoguru", "domlur", "dommasandra", "epip zone", "electronic city", "electronic city phase ii", "electronics city phase 1", "frazer town", "gm palaya", "garudachar palya", "giri nagar", "gollarapalya hosahalli", "gottigere", "green glen layout", "gubbalala", "gunjur", "hal 2nd stage", "hbr layout", "hrbr layout", "hsr layout", "haralur road", "harlur", "hebbal", "hebbal kempapura", "hegde nagar", "hennur", "hennur road", "hoodi", "horamavu agara", "horamavu banaswadi", "hormavu", "hosa road", "hosakerehalli", "hoskote", "hosur road", "hulimavu", "isro layout", "itpl", "iblur village", "indira nagar", "jp nagar", "jakkur", "jalahalli", "jalahalli east", "jigani", "judicial layout", "kr puram", "kadubeesanahalli", "kadugodi", "kaggadasapura", "kaggalipura", "kaikondrahalli", "kalena agrahara", "kalyan nagar", "kambipura", "kammanahalli", "kammasandra", "kanakapura", "kanakpura road", "kannamangala", "karuna nagar", "kasavanhalli", "kasturi nagar", "kathriguppe", "kaval byrasandra", "kenchenahalli", "kengeri", "kengeri satellite town", "kereguddadahalli", "kodichikkanahalli", "kodigehaali", "kodigehalli", "kodihalli", "kogilu", "konanakunte", "koramangala", "kothannur", "kothanur", "kudlu", "kudlu gate", "kumaraswami layout", "kundalahalli", "lb shastri nagar", "laggere", "lakshminarayana pura", "lingadheeranahalli", "magadi road", "mahadevpura", "mahalakshmi layout", "mallasandra", "malleshpalya", "malleshwaram", "marathahalli", "margondanahalli", "marsur", "mico layout", "munnekollal", "murugeshpalya", "mysore road", "ngr layout", "nri layout", "nagarbhavi", "nagasandra", "nagavara", "nagavarapalya", "narayanapura", "neeladri nagar", "nehru nagar", "ombr layout", "old airport road", "old madras road", "padmanabhanagar", "pai layout", "panathur", "parappana agrahara", "pattandur agrahara", "poorna pragna layout", "prithvi layout", "r.t. nagar", "rachenahalli", "raja rajeshwari nagar", "rajaji nagar", "rajiv nagar", "ramagondanahalli", "ramamurthy nagar", "rayasandra", "sahakara nagar", "sanjay nagar", "sarakki nagar", "sarjapur", "sarjapur  road", "sarjapura - attibele road", "sector 2 hsr layout", "sector 7 hsr layout", "seegehalli", "shampura", "shivaji nagar", "singasandra", "somasundara palya", "sompura", "sonnenahalli", "subramanyapura", "sultan palaya", "tc palaya", "talaghattapura", "thanisandra", "thigalarapalya", "thubarahalli", "thyagaraja nagar", "tindlu", "tumkur road", "ulsoor", "uttarahalli", "varthur", "varthur road", "vasanthapura", "vidyaranyapura", "vijayanagar", "vishveshwarya layout", "vishwapriya layout", "vittasandra", "whitefield", "yelachenahalli", "yelahanka", "yelahanka new town", "yelenahalli", "yeshwanthpur"]
location_encoder = OneHotEncoder(categories=[all_locations], sparse_output=False)
location_encoder.fit(np.array(all_locations).reshape(-1, 1))

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        email = request.form['email']
        password = request.form['password']
        user = Users.find_one({'email': email})
        if user and 'password' in user and check_password_hash(user['password'], password):
            return redirect(url_for('home'))
        else:
            flash('Invalid email or password')
    return render_template('login.html')

@app.route('/', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        full_name = request.form['full_name']
        email = request.form['email']
        password = request.form['password']
        phone_number = request.form['phone_number']
        date_of_birth = request.form['date_of_birth']
        gender = request.form['gender']
        user = Users.find_one({'email': email})
        if user:
            flash('Email already exists')
        else:
            new_user = {
                'full_name': full_name,
                'email': email,
                'password': generate_password_hash(password),
                'phone_number': phone_number,
                'date_of_birth': date_of_birth,
                'gender': gender
            }
            Users.insert_one(new_user)
            return redirect(url_for('login'))
    return render_template('index.html')

@app.route('/home', methods=['GET', 'POST'])
def home():
    selected_application = 'Roofing'
    all_foundations = filter_material(selected_application, 'foundation')
    return render_template('home.html',tables=[all_foundations.to_html(classes='data')], titles=all_foundations.columns.values)

@app.route('/calculator', methods=['GET', 'POST'])
def calculator():
    with open('locations.json') as f:
        locations = json.load(f)
    if request.method == 'POST':
        location = request.form['locationSelect']
        bhk = int(request.form.get('bhk'))
        bathrooms = int(request.form.get('bathrooms'))
        square_feet = float(request.form.get('squareFeet'))
        location_encoded = location_encoder.transform(np.array([location]).reshape(-1, 1))
        input_data = np.concatenate([location_encoded, [[bathrooms, square_feet, bhk]]], axis=1)

        if not location or not bhk or not bathrooms or not square_feet:
            return "Error: Missing form fields", 400
        prediction = model.predict(input_data)
        prediction=float(prediction[0])
        return jsonify({'prediction': prediction})
    return render_template('calculator.html',locations=locations)

@app.route('/bid', methods=['GET', 'POST'])
def bidsphere():
    return render_template('bid.html')

@app.route('/schemes', methods=['GET', 'POST'])
def schemes():
    return render_template('s-c-h-e-m-e.html')

@app.route('/engineers', methods=['GET', 'POST'])
def engineers():
    return render_template('ENGINEER.html')

@app.route('/profile', methods=['GET', 'POST'])
def profile():
    if request.method == 'POST':
        user_id = request.form['user_id']
        user = Users.find_one({'_id': user_id})
        user['name'] = request.form['name']
        user['email'] = request.form['email']
        user['mobile_number'] = request.form['mobile_number']
        user['date_of_birth'] = request.form['date_of_birth']
        Users.update_one({'_id': user_id}, {'$set': user})
        flash('Profile updated successfully')
        return redirect(url_for('profile'))
    user_id = request.args.get('user_id')
    user = Users.find_one({'_id': user_id})
    return render_template('profile.html',user=user)



if __name__ == '__main__':
    app.run(debug=True)