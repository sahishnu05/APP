from flask import Flask, render_template, request, jsonify, redirect, url_for
from flask_pymongo import PyMongo
import pandas as pd
import numpy as np
import json
import os
import joblib
from datetime import datetime
from bson.objectid import ObjectId
import math
from werkzeug.security import generate_password_hash, check_password_hash
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import sigmoid_kernel

app = Flask(__name__,template_folder='templates', static_folder='static')

# MongoDB Configuration
app.config["MONGO_URI"] = "mongodb://localhost:27017/construction_db"
mongo = PyMongo(app)

# Database collections - Separate collections for Users and Engineers
Users = mongo.db.users
Engineers = mongo.db.engineers  # New separate collection for engineers
Projects = mongo.db.projects
Bids = mongo.db.bids

# Load or create locations list
try:
    with open('location_columns.json', 'r') as f:
        locations = json.load(f)
except FileNotFoundError:
    # Default locations if file not found
    locations = ["1st Phase JP Nagar", "Indira Nagar", "Whitefield", "Electronic City", 
                "HSR Layout", "Koramangala", "Bannerghatta Road", "Marathahalli"]
    with open('location_columns.json', 'w') as f:
        json.dump(locations, f)

# ML Model Implementation
def predict_price(location, sqft, bath, bhk):
    """Predict house price based on location, square feet, bathrooms and BHK"""
    try:
        # Try to load the model if it exists
        lr_clf = joblib.load('bengaluru_house_price_model.pkl')
        
        # Find the location index
        loc_index = locations.index(location) if location in locations else -1
        
        # Create feature vector
        x = np.zeros(len(locations) + 3)
        x[0] = float(sqft)
        x[1] = int(bath)
        x[2] = int(bhk)
        if loc_index >= 0:
            x[loc_index + 3] = 1
            
        # Predict price
        return int(lr_clf.predict([x])[0])
    except:
        # Fallback prediction if model loading fails
        # Simple formula based on the data patterns
        sqft_price = 5000  # Base price per sqft in rupees
        if location in ["Indira Nagar", "Koramangala", "HSR Layout"]:
            sqft_price = 12000  # Premium areas
        elif location in ["Whitefield", "Electronic City", "Marathahalli"]:
            sqft_price = 7000  # Mid-range areas
            
        base_price = (float(sqft) * sqft_price) / 100000  # Convert to lakhs
        bhk_factor = 1.0 + (0.1 * (int(bhk) - 1))  # 10% increase per additional bedroom
        bath_factor = 1.0 + (0.05 * (int(bath) - int(bhk)))  # Premium for extra bathrooms
        
        return round(base_price * bhk_factor * max(1.0, bath_factor), 2)

# Bidding Algorithm
def compute_score(experience, num_projects, ratings, bid_price,
                  max_experience, max_projects, P_user,
                  w_e=0.30, w_p=0.20, w_b=0.25, w_r=0.25, C=5, epsilon=1e-6):
   
    # Logarithmic Normalization for Experience & Project Count
    S_E = math.log(1 + experience) / math.log(1 + max_experience + epsilon)
    S_P = math.log(1 + num_projects) / math.log(1 + max_projects + epsilon)
   
    # Normalize Bid Price (Higher price = lower score)
    S_B = max(0, 1 - ((bid_price - P_user) / (P_user * 2 + epsilon)))  
   
    # Compute Rating Score using only projects from our website
    num_rated_projects = len(ratings)  # Only projects from our platform
    avg_rating = sum(ratings) / (num_rated_projects + epsilon) if num_rated_projects > 0 else 0
    S_R = min(1, avg_rating * (num_rated_projects / (num_rated_projects + C)))  # Clamped to 1

    # Normalize Weights (Ensures total = 1)
    total_weight = w_e + w_p + w_b + w_r
    w_e /= total_weight
    w_p /= total_weight
    w_b /= total_weight
    w_r /= total_weight

    # Compute Final Score (Clamped to [0,1])
    final_score = min(1, (w_e * S_E) + (w_p * S_P) + (w_b * S_B) + (w_r * S_R))
   
    return final_score

def get_recommendations(material_type, application):
    """Get recommendations for a specific material type and application."""
    try:
        # Load appropriate dataset and determine cost column based on material type
        cost_column = "Cost (per unit)"  # Default cost column name
        if "cement" in material_type.lower():
            filename = "cement.csv"
            cost_column = "Cost (per bag)"
        elif "rebar" in material_type.lower():
            filename = "rebar.csv"
            cost_column = "Cost (per unit)"
        elif "brick" in material_type.lower():
            filename = "bricks.csv"
            cost_column = "Cost (per piece)"
        elif "sand" in material_type.lower():
            filename = "sand.csv"
            cost_column = "Cost (per unit)"
        else:
            filename = "cement.csv"
            cost_column = "Cost (per bag)"
        
        # Add error handling for loading CSV
        try:
            df = pd.read_csv(filename)
            
            # Fix column names - ensure consistent naming
            df.columns = [col.strip() for col in df.columns]
            
            # Map standard columns to actual column names in the dataset
            column_mapping = {
                'Material': 'Material',
                'Application': 'Application',
                'Cost': cost_column,  # This will vary by material type
                'Durability': 'Durability',
                'Ratings': 'Ratings',
                'Review': 'Review'
            }
            
            # Check which columns actually exist in the dataset
            actual_columns = {}
            for std_col, file_col in column_mapping.items():
                if file_col in df.columns:
                    actual_columns[std_col] = file_col
                else:
                    # Use standard column name if file column doesn't exist
                    actual_columns[std_col] = std_col
                    df[std_col] = "Not specified" if std_col != 'Ratings' else 0
            
            # Force numeric columns to be numeric, coercing errors to NaN
            if cost_column in df.columns:
                df[cost_column] = pd.to_numeric(df[cost_column], errors='coerce')
            if 'Ratings' in df.columns:
                df['Ratings'] = pd.to_numeric(df['Ratings'], errors='coerce')
                
            # Fill any NaN values with defaults
            default_values = {
                'Material': 'Unknown Material',
                'Application': 'General Use',
                cost_column: 0.0,
                'Durability': 'Standard',
                'Ratings': 3.0,
                'Review': 'No review available'
            }
            
            for col, default in default_values.items():
                if col in df.columns:
                    df[col] = df[col].fillna(default)
            
            # Filter by application if specified
            if application and application != "All":
                app_df = df[df['Application'] == application]
                if app_df.empty:
                    # Return empty DataFrame with correct column structure
                    empty_df = pd.DataFrame(columns=list(actual_columns.values()))
                    # Add the cost_column as metadata
                    empty_df.attrs['cost_column'] = cost_column
                    return empty_df
            else:
                app_df = df
            
            # Handle empty dataframe case
            if app_df.empty:
                empty_df = pd.DataFrame(columns=list(actual_columns.values()))
                empty_df.attrs['cost_column'] = cost_column
                return empty_df
            
            # Add cost_column as metadata to be accessed in the template
            result_df = app_df.sort_values(by='Ratings', ascending=False)
            result_df.attrs['cost_column'] = cost_column
            
            # Return all materials sorted by ratings (descending)
            return result_df
            
        except Exception as e:
            print(f"Error loading dataset {filename}: {str(e)}")
            # Create an empty DataFrame with proper columns as fallback
            empty_df = pd.DataFrame(columns=['Material', 'Application', cost_column, 
                                            'Durability', 'Ratings', 'Review'])
            empty_df.attrs['cost_column'] = cost_column
            return empty_df
    except Exception as e:
        print(f"Recommendation error: {str(e)}")
        return pd.DataFrame(columns=['Material', 'Application', 'Cost', 
                                    'Durability', 'Ratings', 'Review'])

def get_materials_for_type_and_application(material_type, application=None):
    """Get all materials available for a specific material type and application"""
    try:
        # Load appropriate dataset based on material type
        if "cement" in material_type.lower():
            filename = "cement.csv"
        elif "rebar" in material_type.lower():
            filename = "rebar.csv"
        elif "brick" in material_type.lower():
            filename = "bricks.csv"
        elif "sand" in material_type.lower():
            filename = "sand.csv"
        else:
            filename = "cement.csv"
        
        df = pd.read_csv(filename)
        
        # Filter by application if specified
        if application and application != "All":
            filtered_df = df[df['Application'] == application]
            if filtered_df.empty:
                return []
            materials = filtered_df['Material'].unique().tolist()
        else:
            materials = df['Material'].unique().tolist()
        
        return sorted(materials)
    except Exception as e:
        print(f"Error getting materials: {str(e)}")
        return []

@app.route('/home/<user_id>', methods=['GET', 'POST'])
def home(user_id):
    user = Users.find_one({'_id': ObjectId(user_id)})
    if not user:
        return redirect(url_for('login_user'))

    # Initialize with default values
    default_project = {'_id': 'placeholder'}
    recommendations = []
    selected_application = "All"  # Default to "All"
    material_type = "cement"  # Default material type

    # Get all available applications from all datasets and organize by material type
    try:
        material_datasets = {
            "cement": "cement.csv",
            "rebar": "rebar.csv",
            "bricks": "bricks.csv",
            "sand": "sand.csv"
        }
        
        all_applications = []
        applications_by_material = {}
        
        for mat_type, dataset_name in material_datasets.items():
            try:
                df = pd.read_csv(dataset_name)
                # Fix column names and data types
                df.columns = [col.strip() for col in df.columns]
                
                if 'Application' in df.columns:
                    # Clean application values - strip whitespace
                    df['Application'] = df['Application'].astype(str).str.strip()
                    mat_applications = df['Application'].unique().tolist()
                    applications_by_material[mat_type] = ["All"] + sorted(mat_applications)
                    all_applications.extend(mat_applications)
                else:
                    applications_by_material[mat_type] = ["All"]
            except Exception as e:
                print(f"Error loading dataset {dataset_name}: {str(e)}")
                applications_by_material[mat_type] = ["All"]
        
        # Remove duplicates and sort
        all_applications = ["All"] + sorted(list(set(all_applications)))
        
    except Exception as e:
        print(f"Error loading datasets: {str(e)}")
        all_applications = ["All"]
        applications_by_material = {
            "cement": ["All"],
            "rebar": ["All"],
            "bricks": ["All"],
            "sand": ["All"]
        }

    if request.method == 'POST':
        material_type = request.form.get('material_type', 'cement')
        selected_application = request.form.get('application', 'All')
        
        # Get recommendations based on selection
        recommendations_df = get_recommendations(
            material_type=material_type,
            application=selected_application
        )
        
        if not recommendations_df.empty:
            # Convert to dict for template use
            recommendations = recommendations_df.to_dict('records')
            # Pass the cost column name to the template
            cost_column = getattr(recommendations_df.attrs, 'cost_column', 'Cost')
        else:
            recommendations = []
            # Set default cost column based on material type
            if "cement" in material_type.lower():
                cost_column = "Cost (per bag)"
            elif "brick" in material_type.lower():
                cost_column = "Cost (per piece)"
            else:
                cost_column = "Cost (per unit)"
    else:
        # On initial page load, get recommendations for the default selections
        recommendations_df = get_recommendations('cement', 'All')
        if not recommendations_df.empty:
            recommendations = recommendations_df.to_dict('records')
            cost_column = getattr(recommendations_df.attrs, 'cost_column', 'Cost (per bag)')
        else:
            recommendations = []
            cost_column = "Cost (per bag)"  # Default for cement

    return render_template('home.html',
                         user=user,
                         project=default_project,
                         applications=all_applications,
                         applications_by_material=applications_by_material,
                         recommendations=recommendations,
                         selected_application=selected_application,
                         material_type=material_type,
                         cost_column=cost_column)  # Pass cost column to template

def check_password(stored_password, provided_password):
    return check_password_hash(stored_password, provided_password)

# Modified registration route for regular users
@app.route('/register/user', methods=['GET', 'POST'])
def register_user():
    if request.method == 'POST':
        # Get form data
        full_name = request.form['full_name']
        email = request.form['email']
        password = request.form['password']
        phone_number = request.form['phone_number']
        date_of_birth = request.form['date_of_birth']
        gender = request.form['gender']
        
        # Check if user already exists in Users collection
        existing_user = Users.find_one({'email': email})
        if existing_user:
            return redirect(url_for('register_user'))
            
        # Hash the password
        hashed_password = generate_password_hash(password)
        
        # Create new user
        new_user = {
            'full_name': full_name,
            'email': email,
            'password': hashed_password,
            'phone_number': phone_number,
            'date_of_birth': date_of_birth,
            'gender': gender,
            'created_at': datetime.utcnow()
        }
        
        # Insert user into Users collection
        user_id = Users.insert_one(new_user).inserted_id
        return redirect(url_for('login_user'))
    
    return render_template('index.html')

@app.route('/register_engineer', methods=['GET', 'POST'])
def register_engineer():
    if request.method == 'POST':
        # Get form data from the engineer registration form
        full_name = request.form['fullName']
        email = request.form['email']
        password = request.form['password']
        college_name = request.form['collegeName']
        degree = request.form['degree']
        field_of_study = request.form.get('fieldOfStudy', '')
        years_experience = int(request.form['experience'])
        project_count = int(request.form['projectCount'])
        country = request.form['country']
        city = request.form['city']
        
        # Handle resume upload
        resume = request.files['resume']
        resume_path = f'uploads/resumes/{resume.filename}'
        os.makedirs(os.path.dirname(resume_path), exist_ok=True)
        resume.save(resume_path)
        
        # Check if engineer already exists in Engineers collection
        existing_engineer = Engineers.find_one({'email': email})
        if existing_engineer:
            return redirect(url_for('register_engineer'))
            
        # Hash the password
        hashed_password = generate_password_hash(password)
        
        # Create new engineer
        new_engineer = {
            'full_name': full_name,
            'email': email,
            'password': hashed_password,
            'college_name': college_name,
            'degree': degree,
            'field_of_study': field_of_study,
            'years_experience': years_experience,
            'projects_completed': project_count,
            'country': country,
            'city': city,
            'resume_path': resume_path,
            'ratings': [],
            'joined_date': datetime.utcnow(),
            'profile_complete': True
        }
        
        # Insert engineer into Engineers collection
        engineer_id = Engineers.insert_one(new_engineer).inserted_id
        print("Registration successful!")        
        return redirect(url_for('engineer_dashboard', engineer_id=str(engineer_id)))
    
    return render_template('engineer_register.html')

# Session-less login handlers - Modified for separate collections
@app.route('/login/engineer', methods=['GET', 'POST'])
def login_engineer():
    if request.method == 'POST':
        email = request.form['email']
        password = request.form['password']
        engineer = Engineers.find_one({'email': email})
            
        if engineer and check_password(engineer['password'], password):
            return redirect(url_for('engineer_dashboard', engineer_id=engineer['_id']))
        else:
            return render_template('engineer_login.html', error='Invalid email or password')
    return render_template('engineer_login.html')

@app.route('/login/user', methods=['GET', 'POST'])
def login_user():
    if request.method == 'POST':
        email = request.form['email']
        password = request.form['password']
        user = Users.find_one({'email': email})
        
        if user and check_password(user['password'], password):
            return redirect(url_for('home', user_id=user['_id']))
        else:
            return render_template('login.html', error='Invalid email or password')
    return render_template('login.html')

@app.route('/logout')
def logout():
    return redirect(url_for('login_user'))

@app.route('/')
def index():
    return render_template('main.html')

# Dashboard route with engineer ID
@app.route('/engineer_dashboard/<engineer_id>')
def engineer_dashboard(engineer_id):
    try:
        engineer = Engineers.find_one({'_id': ObjectId(engineer_id)})
        if not engineer:
            return redirect(url_for('login_engineer'))
        
        # Fetch open projects
        open_projects = list(Projects.find({'status': 'open'}))
        
        # Fetch user information for each project
        projects_users = []
        projects_with_users = []
        for project in open_projects:
            user = Users.find_one({'_id': project['user_id']})
            projects_users.append(user)
            projects_with_users.append({
                'project': project,
                'user': user
            })
        
        active_bids = list(Bids.find({'engineer_id': ObjectId(engineer_id)}))
        return render_template('engineerspage.html',
                             engineer=engineer,
                             open_projects=open_projects,
                             user=projects_users,
                             active_bids=active_bids)    
    except Exception as e:
        print(f"Dashboard error: {str(e)}")
        return "Error loading dashboard", 500


@app.route('/profile/<user_id>')
def profile(user_id):
    user = Users.find_one({'_id': ObjectId(user_id)})
    if not user:
        return redirect(url_for('login_user')) 
    project = Projects.find_one({'user_id': ObjectId(user_id)})  # Adjust query as needed

    if not project:
        print("No project found for this user.")
        project = {'_id': 'placeholder'}
    return render_template('profile.html', user=user, user_id=user_id, project=project)


@app.route('/schemes/<user_id>')
def schemes(user_id):
    user = Users.find_one({'_id': ObjectId(user_id)})
    if not user:
        return redirect(url_for('login_user')) 
    project = Projects.find_one({'user_id': ObjectId(user_id)})
    if not project:
        print("No project found for this user.")
        project = None 
    return render_template('s-c-h-e-m-e.html', user=user, user_id=user_id, project=project)

# Calculator with user ID
@app.route('/calculator', methods=['GET', 'POST'])
@app.route('/calculator/<user_id>', methods=['GET', 'POST'])
def calculator(user_id=None):
    """Calculator route that works with or without a user ID"""
    user = None
    if user_id:
        try:
            user = Users.find_one({'_id': ObjectId(user_id)})
        except:
            # Invalid user ID format or user not found
            pass
            
    # Default project for context in template
    project = None
    if user:
        project = Projects.find_one({'user_id': ObjectId(user_id)})
        
    if request.method == 'POST':
        try:
            location = request.form['locationSelect']
            bhk = int(request.form['bhk'])
            bathrooms = int(request.form['bathrooms'])
            sqft = float(request.form['squareFeet'])
            
            # Validate inputs
            if sqft < 300 or sqft > 10000:
                return jsonify({'error': 'Square feet should be between 300 and 10000'})
                
            if bathrooms < 1 or bathrooms > 10:
                return jsonify({'error': 'Bathrooms should be between 1 and 10'})
                
            if bhk < 1 or bhk > 10:
                return jsonify({'error': 'BHK should be between 1 and 10'})
                
            # Predict price
            prediction = predict_price(location, sqft, bathrooms, bhk)
            formatted_prediction = f"₹{prediction:.2f} Lakh"
            
            return jsonify({'prediction': formatted_prediction})
        except Exception as e:
            return jsonify({'error': str(e)})
    
    return render_template('calculator.html', locations=locations, user=user, project=project)

# Helper function to rank engineers for a project
def rank_engineers_for_project(project, bids):
    """Rank engineers based on their bids and experience"""
    try:
        # Find max experience and projects among all engineers
        all_engineers = list(Engineers.find())
        max_experience = max([eng.get('years_experience', 0) for eng in all_engineers]) if all_engineers else 10
        max_projects = max([eng.get('projects_completed', 0) for eng in all_engineers]) if all_engineers else 20
        
        # User's desired budget
        P_user = float(project.get('desired_budget', 0))
        
        # Calculate scores for each bid
        ranked_bids = []
        for bid in bids:
            engineer_id = bid.get('engineer_id')
            engineer = Engineers.find_one({'_id': engineer_id})
            
            if engineer:
                experience = engineer.get('years_experience', 0)
                num_projects = engineer.get('projects_completed', 0)
                ratings = engineer.get('ratings', [])
                bid_price = float(bid.get('bid_amount', 0))
                
                # Calculate score using the bidding algorithm
                score = compute_score(
                    experience=experience,
                    num_projects=num_projects,
                    ratings=ratings,
                    bid_price=bid_price,
                    max_experience=max_experience,
                    max_projects=max_projects,
                    P_user=P_user
                )
                
                # Format for display
                ranked_bids.append({
                    '_id': bid.get('_id'),
                    'engineer_id': engineer_id,
                    'engineer': engineer,
                    'bid_amount': bid_price,
                    'score': score,
                    'score_percentage': int(score * 100)
                })
        
        # Sort by score (highest first)
        return sorted(ranked_bids, key=lambda x: x['score'], reverse=True)
        
    except Exception as e:
        print(f"Ranking error: {str(e)}")
        return []

@app.route('/bidsphere/<user_id>', methods=['GET', 'POST'])
def bidsphere(user_id):
    # Verify user exists
    user = Users.find_one({'_id': ObjectId(user_id)})
    if not user:
        return redirect(url_for('login_user'))
    
    # Find any existing project for this user or create placeholder
    project = Projects.find_one({'user_id': ObjectId(user_id)})
    if not project:
        project = {'_id': 'placeholder'}
    
    if request.method == 'POST':
        try:
            # Get form data
            estimated_price = float(request.form['estimated_price'])
            area = float(request.form['area'])
            district = request.form['district']
            instructions = request.form.get('instructions', '')
            
            # Handle file uploads
            property_document = request.files['property_document']
            property_doc_path = None
            if property_document:
                filename = f"{user_id}_{datetime.utcnow().strftime('%Y%m%d%H%M%S')}_property.pdf"
                upload_dir = os.path.join('static', 'uploads', 'documents')
                os.makedirs(upload_dir, exist_ok=True)
                property_doc_path = os.path.join(upload_dir, filename)
                property_document.save(property_doc_path)
            
            plan_proposal = request.files.get('plan_proposal')
            plan_path = None
            if plan_proposal and plan_proposal.filename:
                filename = f"{user_id}_{datetime.utcnow().strftime('%Y%m%d%H%M%S')}_plan.pdf"
                plan_path = os.path.join(upload_dir, filename)
                plan_proposal.save(plan_path)
            
            # Create new project for bidding
            new_project = {
                'user_id': ObjectId(user_id),
                'project_title': f"Property Development in {district.capitalize()}",
                'property_type': 'residential',
                'construction_area': area,
                'location': district,
                'description': instructions,
                'desired_budget': estimated_price,
                'timeline_months': 12,  # Default timeline
                'status': 'open',
                'creation_date': datetime.utcnow(),
                'property_document': property_doc_path,
                'plan_proposal': plan_path,
                'selected_engineer_id': None
            }
            
            # Insert the project
            project_id = Projects.insert_one(new_project).inserted_id
            
            # Redirect to engineer listing page for this new project
            return redirect(url_for('engineerlist', project_id=project_id, user_id=user_id))
            
        except Exception as e:
            print(f"Error creating bidding project: {str(e)}")
            return render_template('bid.html', 
                                  user=user, 
                                  user_id=user_id,
                                  project=project,
                                  error="An error occurred while submitting your bid request")
    
    # GET request - display the form
    return render_template('bid.html', user=user, user_id=user_id, project=project)

@app.route('/engineerlist/<project_id>/<user_id>')
def engineerlist(project_id, user_id):
    """Display lists of engineers for projects grouped by status"""
    user = Users.find_one({'_id': ObjectId(user_id)})
    if not user:
        return redirect(url_for('login_user'))
    
    try:
        # Handle the placeholder case - find if user has any projects
        if project_id == 'placeholder':
            # Check if user has any projects
            user_projects = list(Projects.find({'user_id': ObjectId(user_id)}))
            if not user_projects:
                # No projects yet, show empty engineer list
                return render_template(
                    'engineerlist.html',
                    user=user,
                    user_id=user_id,
                    project={'_id': 'placeholder', 'status': 'none'},
                    engineers=[],
                    current_date=datetime.now(),
                    no_projects=True
                )
        
        # Get all user projects by status
        open_projects = list(Projects.find({
            'user_id': ObjectId(user_id),
            'status': 'open'
        }))
        
        awarded_projects = list(Projects.find({
            'user_id': ObjectId(user_id),
            'status': 'awarded'
        }))
        
        completed_projects = list(Projects.find({
            'user_id': ObjectId(user_id),
            'status': 'completed'
        }))
        
        # Process open projects - add engineers for each
        for project in open_projects:
            bids = list(Bids.find({'project_id': project['_id']}))
            if bids:
                ranked_bids = rank_engineers_for_project(project, bids)
                project['engineers'] = [
                    {
                        '_id': bid['engineer_id'],
                        'full_name': bid['engineer']['full_name'],
                        'bid_amount': bid['bid_amount'],
                        'score_percentage': bid['score_percentage']
                    } for bid in ranked_bids
                ]
            else:
                project['engineers'] = []
                
        # Process awarded projects - add selected engineer info
        for project in awarded_projects:
            if 'selected_engineer_id' in project:
                engineer = Engineers.find_one({'_id': project['selected_engineer_id']})
                bid = Bids.find_one({
                    'project_id': project['_id'],
                    'engineer_id': project['selected_engineer_id']
                })
                
                if engineer and bid:
                    project['engineer'] = {
                        '_id': engineer['_id'],
                        'full_name': engineer['full_name'],
                        'bid_amount': bid['bid_amount']
                    }
                    
        # Process completed projects - add rating info
        for project in completed_projects:
            if 'selected_engineer_id' in project:
                engineer = Engineers.find_one({'_id': project['selected_engineer_id']})
                bid = Bids.find_one({
                    'project_id': project['_id'],
                    'engineer_id': project['selected_engineer_id']
                })
                
                if engineer and bid:
                    project['engineer'] = {
                        '_id': engineer['_id'],
                        'full_name': engineer['full_name'],
                        'bid_amount': bid['bid_amount']
                    }
                    
                    # Check if this project has been rated
                    project['is_rated'] = 'rating' in project
                    if project['is_rated']:
                        project['rating'] = project['rating']
                    
        # Get current date for footer
        current_date = datetime.now()
        
        return render_template(
            'engineerlist.html',
            user=user,
            user_id=user_id,
            open_projects=open_projects,
            awarded_projects=awarded_projects,
            completed_projects=completed_projects,
            current_date=current_date
        )
    except Exception as e:
        print(f"Error loading engineer list: {str(e)}")
        return "An error occurred", 500

@app.route('/mark_project_complete/<project_id>/<user_id>', methods=['POST'])
def mark_project_complete(project_id, user_id):
    """Mark a project as completed"""
    try:
        # Update project status to completed
        Projects.update_one(
            {'_id': ObjectId(project_id)},
            {'$set': {'status': 'completed'}}
        )
        return redirect(url_for('engineerlist', project_id='placeholder', user_id=user_id))
    except Exception as e:
        print(f"Error marking project as complete: {str(e)}")
        return "An error occurred", 500

@app.route('/rate_engineer/<project_id>/<engineer_id>/<user_id>', methods=['POST'])
def rate_engineer(project_id, engineer_id, user_id):
    """Rate an engineer for a completed project"""
    try:
        rating = int(request.form['rating'])
        if rating < 1 or rating > 5:
            return "Invalid rating", 400
            
        # Update project with rating
        Projects.update_one(
            {'_id': ObjectId(project_id)},
            {'$set': {'rating': rating}}
        )
        
        # Add rating to engineer's ratings array
        Engineers.update_one(
            {'_id': ObjectId(engineer_id)},
            {'$push': {'ratings': rating}}
        )
        
        return redirect(url_for('engineerlist', project_id='placeholder', user_id=user_id))
    except Exception as e:
        print(f"Error rating engineer: {str(e)}")
        return "An error occurred", 500

@app.route('/accept_engineer/<project_id>/<engineer_id>/<user_id>', methods=['POST'])
def accept_engineer(project_id, engineer_id, user_id):
    """Accept an engineer's bid for a project"""
    try:
        # Update project with selected engineer
        Projects.update_one(
            {'_id': ObjectId(project_id)},
            {'$set': {'selected_engineer_id': ObjectId(engineer_id), 'status': 'awarded'}}
        )
        return redirect(url_for('home', user_id=user_id))
    except Exception as e:
        print(f"Error accepting engineer: {str(e)}")
        return "An error occurred", 500

@app.route('/accept_bid/<project_id>/<engineer_id>/<user_id>', methods=['POST'])
def accept_bid(project_id, engineer_id, user_id):
    try:
        # Update project with selected engineer
        Projects.update_one(
            {'_id': ObjectId(project_id)},
            {'$set': {
                'selected_engineer_id': ObjectId(engineer_id),
                'status': 'awarded'
            }}
        )
        
        # Update all bids status
        Bids.update_many(
            {'project_id': ObjectId(project_id)},
            {'$set': {'status': 'closed'}}
        )
        
        return redirect(url_for('engineerlist', 
                              project_id=project_id, 
                              user_id=user_id))
    except Exception as e:
        print(f"Error accepting bid: {str(e)}")
        return "An error occurred", 500

@app.route('/project/<project_id>')
def get_project(project_id):
    """API endpoint to get project details for engineers"""
    try:
        project = Projects.find_one({'_id': ObjectId(project_id)})
        if not project:
            return jsonify({'error': 'Project not found'}), 404
            
        # Convert ObjectId to string for JSON serialization
        project_data = {
            '_id': str(project['_id']),
            'user_id': str(project['user_id']),
            'project_title': project.get('project_title', ''),
            'location': project.get('location', ''),
            'construction_area': project.get('construction_area', 0),
            'desired_budget': project.get('desired_budget', 0),
            'description': project.get('description', ''),
            'status': project.get('status', 'open'),
        }
        
        return jsonify(project_data)
    except Exception as e:
        print(f"Error fetching project: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/submit_engineer_bid/<project_id>/<engineer_id>', methods=['POST'])
def submit_engineer_bid(project_id, engineer_id):
    """Submit a bid for a project as an engineer"""
    try:
        price = float(request.form['price'])
        
        # Check if engineer already has a bid for this project
        existing_bid = Bids.find_one({
            'project_id': ObjectId(project_id),
            'engineer_id': ObjectId(engineer_id)
        })
        
        if existing_bid:
            # Update existing bid
            Bids.update_one(
                {'_id': existing_bid['_id']},
                {'$set': {'bid_amount': price, 'updated_at': datetime.utcnow()}}
            )
        else:
            # Create new bid
            new_bid = {
                'project_id': ObjectId(project_id),
                'engineer_id': ObjectId(engineer_id),
                'bid_amount': price,
                'created_at': datetime.utcnow(),
                'status': 'pending'
            }
            Bids.insert_one(new_bid)
        
        # Redirect back to engineer dashboard
        return redirect(url_for('engineer_dashboard', engineer_id=engineer_id))
    except Exception as e:
        print(f"Error submitting bid: {str(e)}")
        return "An error occurred", 500

# New route to show allotted projects for engineers
@app.route('/allotted_projects/<engineer_id>')
def allotted_projects(engineer_id):
    try:
        engineer = Engineers.find_one({'_id': ObjectId(engineer_id)})
        if not engineer:
            return redirect(url_for('login_engineer'))
        
        # Find all projects where this engineer is selected
        allotted_projects = list(Projects.find({
            'selected_engineer_id': ObjectId(engineer_id),
            'status': {'$in': ['awarded', 'completed']}
        }))
        
        # Add bid amounts to each project
        for project in allotted_projects:
            bid = Bids.find_one({
                'project_id': project['_id'],
                'engineer_id': ObjectId(engineer_id)
            })
            
            if bid:
                project['bid_amount'] = bid['bid_amount']
            else:
                project['bid_amount'] = 'N/A'
                
        return render_template('allotted_projects.html',
                             engineer=engineer,
                             allotted_projects=allotted_projects)
    except Exception as e:
        print(f"Error loading allotted projects: {str(e)}")
        return "Error loading allotted projects", 500

# Engineer profile page
@app.route('/engineer_profile/<engineer_id>')
def engineer_profile(engineer_id):
    try:
        engineer = Engineers.find_one({'_id': ObjectId(engineer_id)})
        if not engineer:
            return redirect(url_for('login_engineer'))
            
        return render_template('engineer_profile.html', engineer=engineer)
    except Exception as e:
        print(f"Error loading engineer profile: {str(e)}")
        return "Error loading profile", 500

# Route for engineers to mark a project as complete
@app.route('/mark_project_complete_engineer/<project_id>/<engineer_id>', methods=['POST'])
def mark_project_complete_engineer(project_id, engineer_id):
    try:
        # Verify engineer is assigned to this project
        project = Projects.find_one({
            '_id': ObjectId(project_id),
            'selected_engineer_id': ObjectId(engineer_id)
        })
        
        if not project:
            return "Unauthorized", 403
            
        # Update project status
        Projects.update_one(
            {'_id': ObjectId(project_id)},
            {'$set': {'status': 'completed'}}
        )
        
        return redirect(url_for('allotted_projects', engineer_id=engineer_id))
    except Exception as e:
        print(f"Error marking project as complete: {str(e)}")
        return "An error occurred", 500

# Logout route for engineers
@app.route('/logout_engineer')
def logout_engineer():
    return redirect(url_for('login_engineer'))


if __name__ == '__main__':
    app.run(debug=True)
