import random

clusters = {
    'Maths': ['Mathematics Advanced', 'Mathematics Extension 1', 'Mathematics Extension 2', 'Mathematics Standard', 'Software Engineering'],
    'Science': ['Biology', 'Physics'],
    'HSIE': ['Ancient History', 'Biblical Studies', 'Business Studies', 'Commerce', 'Community and Family Studies', 'Economics', 'Geography', 'Legal Studies', 'School of Languages', 'Society and Culture'],
    'TAS': ['Design and Technology', 'Engineering Studies', 'Food Technology', 'Hospitality', 'Industrial Technology Multimedia', 'Industrial Technology Timber'],
    'English': ['English Advanced', 'English Extension', 'English Standard'],
    'PDHPE': ['Health and Movement Science'],
    'Visual and Performing Arts': ['Drama', 'Music', 'Visual Arts']
}

def generate_row(i):
    # Base profiles
    profile = random.choice(['Maths', 'Science', 'HSIE', 'TAS', 'English', 'PDHPE', 'Visual and Performing Arts'])
    
    # Generate scores based on profile
    q1 = random.randint(7, 10) if profile in ['Maths', 'Science'] else random.randint(1, 6)
    q2 = random.randint(4, 5) if profile in ['English', 'HSIE'] else random.randint(1, 3)
    q3 = 1 if profile in ['TAS'] else random.randint(0, 1)
    q4 = random.randint(7, 10) if profile in ['Visual and Performing Arts', 'HSIE'] else random.randint(1, 6)
    q5 = random.randint(4, 5) if profile in ['Science', 'PDHPE'] else random.randint(1, 3)
    q6 = 1 if profile in ['HSIE'] else random.randint(0, 1)
    q7 = random.randint(7, 10) if profile in ['Maths', 'TAS'] else random.randint(1, 6)
    q8 = random.randint(4, 5) if profile in ['Visual and Performing Arts'] else random.randint(1, 3)
    q9 = random.randint(7, 10) if profile in ['HSIE', 'English'] else random.randint(1, 6)
    q10 = 1 if profile in ['Science'] else random.randint(0, 1)
    q11 = random.randint(4, 5) if profile in ['TAS'] else random.randint(1, 3) # Hospitality/Food Tech
    q12 = random.randint(7, 10) if profile in ['Maths', 'Science', 'TAS'] else random.randint(1, 6)
    
    # Pick subjects mostly from their profile
    my_subjects = random.sample(clusters[profile], min(3, len(clusters[profile])))
    other_clusters = [c for c in clusters.keys() if c != profile]
    other_subjs = []
    for _ in range(6 - len(my_subjects)):
        c = random.choice(other_clusters)
        other_subjs.append(random.choice(clusters[c]))
    
    subjects = my_subjects + other_subjs
    random.shuffle(subjects)
    
    # Optional 7th subject
    subj7 = ""
    if random.random() > 0.8:
        c = random.choice(list(clusters.keys()))
        subj7 = random.choice(clusters[c])
        
    satisfaction = random.randint(7, 10)
    
    return f"S{i:03d},{q1},{q2},{q3},{q4},{q5},{q6},{q7},{q8},{q9},{q10},{q11},{q12},{subjects[0]},{subjects[1]},{subjects[2]},{subjects[3]},{subjects[4]},{subjects[5]},{subj7},{satisfaction}"

lines = ["student_id,q1,q2,q3,q4,q5,q6,q7,q8,q9,q10,q11,q12,target_subject_1,target_subject_2,target_subject_3,target_subject_4,target_subject_5,target_subject_6,target_subject_7,satisfaction_score"]
for i in range(1, 101):
    lines.append(generate_row(i))

with open(r"c:\Temp\11-12 SE\12SE\12SE_HSC_subject_selection\training_data.csv", "w", encoding="utf-8") as f:
    f.write("\n".join(lines))
