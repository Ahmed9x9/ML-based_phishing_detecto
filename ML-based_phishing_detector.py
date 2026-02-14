import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import joblib
import warnings
import os
warnings.filterwarnings('ignore')


# Load emails from the CSV files
def load_data_from_excel():
    
    phishing_file = "phishing_emails.csv"
    legitimate_file = "legitimate_emails.csv"
    
    if not os.path.exists(phishing_file):
        print(f"Error: Cannot find '{phishing_file}'")
        return None
    
    if not os.path.exists(legitimate_file):
        print(f"Error: Cannot find '{legitimate_file}'")
        return None
    
    # Load the CSV files
    print("Loading emails...")
    df_phishing = pd.read_csv(phishing_file)    
    df_legitimate = pd.read_csv(legitimate_file)
    
    # Combine
    df = pd.concat([df_phishing, df_legitimate], ignore_index=True)
    
    # Clean the data
    df = df.dropna(subset=['email_text'])
    df = df[df['email_text'].str.len() > 10]
    
    print(f"\nTotal dataset: {len(df):,} emails")
    print(f"- Phishing: {len(df[df['label']==1]):,}")
    print(f"- Legitimate: {len(df[df['label']==0]):,}")
    
    return df


# Save the best model and vectorizer
def save_best_model(name, model, vectorizer):
    
    model_filename = "best_model.joblib"
    vectorizer_filename = "vectorizer.joblib"
    info_filename = "model_info.txt"
    
    joblib.dump(model, model_filename)
    joblib.dump(vectorizer, vectorizer_filename)
    
    with open(info_filename, 'w') as f:
        f.write(name)
    
    print(f"\nModel saved")


# Load the saved model and vectorizer
def load_model_for_classification():
    
    model_filename = "best_model.joblib"
    vectorizer_filename = "vectorizer.joblib"
    info_filename = "model_info.txt"
    
    if not os.path.exists(model_filename) or not os.path.exists(vectorizer_filename):
        print("\nError: No saved model found")
        return None, None, None
    
    model = joblib.load(model_filename)
    vectorizer = joblib.load(vectorizer_filename)
    
    # Load model name
    model_name = "Unknown"
    if os.path.exists(info_filename):
        with open(info_filename, 'r') as f:
            model_name = f.read().strip()
    
    print(f"\nLoaded saved model: {model_name}")
    
    return model, vectorizer, model_name

# Average over 20 times
def run_comparison(df, num_runs=20):
    
    all_results = {
        'Random Forest': {'accuracy': [], 'precision': [], 'recall': [], 'f1_score': [], 'fp': [], 'fn': [], 'tn': [], 'tp': []},
        'SVM': {'accuracy': [], 'precision': [], 'recall': [], 'f1_score': [], 'fp': [], 'fn': [], 'tn': [], 'tp': []},
        'Naive Bayes': {'accuracy': [], 'precision': [], 'recall': [], 'f1_score': [], 'fp': [], 'fn': [], 'tn': [], 'tp': []},
        'Logistic Regression': {'accuracy': [], 'precision': [], 'recall': [], 'f1_score': [], 'fp': [], 'fn': [], 'tn': [], 'tp': []},
        'KNN': {'accuracy': [], 'precision': [], 'recall': [], 'f1_score': [], 'fp': [], 'fn': [], 'tn': [], 'tp': []},
        'Decision Tree': {'accuracy': [], 'precision': [], 'recall': [], 'f1_score': [], 'fp': [], 'fn': [], 'tn': [], 'tp': []}
    }
    
    print(f"\nRunning {num_runs} times...")
    print("")
    
    for i in range(num_runs):
        seed = i + 1
        
        # Shuffle data
        df_shuffled = df.sample(frac=1, random_state=seed).reset_index(drop=True)
        
        # TF-IDF
        vectorizer = TfidfVectorizer(
            max_features=1000,
            stop_words='english',
            ngram_range=(1, 2),
            min_df=2,
            max_df=0.95
        )
        
        X = vectorizer.fit_transform(df_shuffled['email_text'])
        y = df_shuffled['label']
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=seed, stratify=y
        )
        
        # Train and test models
        models = {
            'Random Forest': RandomForestClassifier(n_estimators=100, max_depth=10, random_state=seed),
            'SVM': SVC(kernel='linear', C=1.0, random_state=seed),
            'Naive Bayes': MultinomialNB(alpha=0.1),
            'Logistic Regression': LogisticRegression(max_iter=1000, random_state=seed),
            'KNN': KNeighborsClassifier(n_neighbors=5),
            'Decision Tree': DecisionTreeClassifier(max_depth=10, random_state=seed)
        }
        
        for name, model in models.items():
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            
            cm = confusion_matrix(y_test, y_pred)
            tn, fp, fn, tp = cm.ravel()
            
            all_results[name]['accuracy'].append(accuracy_score(y_test, y_pred))
            all_results[name]['precision'].append(precision_score(y_test, y_pred))
            all_results[name]['recall'].append(recall_score(y_test, y_pred))
            all_results[name]['f1_score'].append(f1_score(y_test, y_pred))
            all_results[name]['fp'].append(fp)
            all_results[name]['fn'].append(fn)
            all_results[name]['tn'].append(tn)
            all_results[name]['tp'].append(tp)
        
        print(f" {i + 1}/{num_runs} complete")
    
    return all_results


def show_results(all_results, num_runs, dataset_size):

    # Calculate averages
    averages = {}
    for name, metrics in all_results.items():
        avg_tp = np.mean(metrics['tp'])
        avg_fn = np.mean(metrics['fn'])
        avg_tn = np.mean(metrics['tn'])
        avg_fp = np.mean(metrics['fp'])
        
        # Phishing recall = TP / (TP + FN)
        phishing_recall = avg_tp / (avg_tp + avg_fn) if (avg_tp + avg_fn) > 0 else 0
        # Legitimate recall = TN / (TN + FP)  
        legitimate_recall = avg_tn / (avg_tn + avg_fp) if (avg_tn + avg_fp) > 0 else 0
        
        averages[name] = {
            'accuracy': np.mean(metrics['accuracy']),
            'precision': np.mean(metrics['precision']),
            'recall': np.mean(metrics['recall']),
            'f1_score': np.mean(metrics['f1_score']),
            'fp': avg_fp,
            'fn': avg_fn,
            'phishing_recall': phishing_recall,
            'legitimate_recall': legitimate_recall,
            'fp_rate': avg_fp / (avg_tn + avg_fp) if (avg_tn + avg_fp) > 0 else 0
        }
    
    print(f"           Average Results Over {num_runs} Runs")
    print("-------------------------------------------------------------------")
    
    for name, data in averages.items():
        print(f"\n{name}")
        print(f"Accuracy: {data['accuracy']*100:.2f}%   Precision: {data['precision']*100:.2f}%   Recall: {data['recall']*100:.2f}%   F1-Score: {data['f1_score']*100:.2f}%")
        print(f"Catches {data['phishing_recall']*100:.1f}% of phishing, flag {data['fp_rate']*100:.1f}% of legitimate as phishing")
        
    # Best model by F1
    best = max(averages.items(), key=lambda x: x[1]['f1_score'])
    best_name = best[0]
    best_data = best[1]
    
    print(f"The Best Model: {best_name} Average F1-Score: {best_data['f1_score']*100:.2f}%")
    
    print("\n           Average Errors per test")
    print("-------------------------------------------------------------------")
    
    for name, data in averages.items():
        print(f"\n{name}:")
        print(f"Average False Positives: {data['fp']:.1f} legitimate emails flagged as phishing")
        print(f"Average False Negatives: {data['fn']:.1f} phishing emails flagged as legitimate")
    
    # Find model with lowest FN
    lowest_fn_model = min(averages.items(), key=lambda x: x[1]['fn'])
    lowest_fp_model = min(averages.items(), key=lambda x: x[1]['fp'])
    
    # Print analysis summary
    print("\n           Analysis Summary")
    print("-------------------------------------------------------------------")
    print(f"\n{best_name} has the best F1-Score {best_data['f1_score']*100:.2f}%")
    print(f"It catches {best_data['phishing_recall']*100:.1f}% of phishing emails while only flags")
    print(f"{best_data['fp_rate']*100:.1f}% of legitimate emails as phishing")
    
    if lowest_fn_model[0] != best_name:
        print(f"\n{lowest_fn_model[0]} has the lowest false negatives {lowest_fn_model[1]['fn']:.1f}")
    
    if lowest_fp_model[0] != best_name:
        print(f"\n{lowest_fp_model[0]} has the lowest false positives {lowest_fp_model[1]['fp']:.1f}")
    
    return best_name, averages


# Train all models and save the best one
def train_and_save_best_model(df):
        
    # Shuffle and prepare data
    df_shuffled = df.sample(frac=1, random_state=42).reset_index(drop=True)
    
    vectorizer = TfidfVectorizer(
        max_features=1000,
        stop_words='english',
        ngram_range=(1, 2),
        min_df=2,
        max_df=0.95
    )
    
    X = vectorizer.fit_transform(df_shuffled['email_text'])
    y = df_shuffled['label']
    
    # Split for evaluation
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Train all models
    models = {
        'Random Forest': RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42),
        'SVM': SVC(kernel='linear', C=1.0, random_state=42, probability=True),
        'Naive Bayes': MultinomialNB(alpha=0.1),
        'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
        'KNN': KNeighborsClassifier(n_neighbors=5),
        'Decision Tree': DecisionTreeClassifier(max_depth=10, random_state=42)
    }
    
    best_f1 = 0
    best_name = None
    best_model = None
    
    print("\nTraining models...")
    for name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        f1 = f1_score(y_test, y_pred)
        print(f"  {name}: F1-Score = {f1*100:.2f}%")
        
        if f1 > best_f1:
            best_f1 = f1
            best_name = name
            best_model = model
    
    # Retrain best model on all data
    print(f"\nBest model: {best_name} F1: {best_f1*100:.2f}%")
    
    best_model_full = models[best_name].__class__(**models[best_name].get_params())
    best_model_full.fit(X, y)
    
    # Save the best model
    save_best_model(best_name, best_model_full, vectorizer)
    
    return best_name, best_model_full, vectorizer


# Classify emails from text file
def classify_from_file(filepath, model, vectorizer, model_name):
    
    if not os.path.exists(filepath):
        print(f"\nError: File '{filepath}' not found")
        return
    
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()
    except Exception as e:
        print(f"\nError reading file: {e}")
        return
    
    if "~" in content:
        emails = [e.strip() for e in content.split("~")]
    else:
        emails = [e.strip() for e in content.split("\n\n")]
    
    # Remove empty emails
    emails = [e for e in emails if e and len(e) > 10]
    
    if not emails:
        print("\nNo valid emails found in file")
        return
    
    classify_emails(emails, model, vectorizer, model_name)


# Classify list of emails
def classify_emails(emails, model, vectorizer, model_name):
    
    X_new = vectorizer.transform(emails)
    
    print(f"           Classification Results using {model_name}")
    print("-------------------------------------------------------------------")
    
    phishing_count = 0
    legitimate_count = 0
    
    for i, email in enumerate(emails):
        print(f"\nEmail {i+1}")
        
        prediction = model.predict(X_new[i:i+1])[0]
        
        if hasattr(model, 'predict_proba'):
            proba = model.predict_proba(X_new[i:i+1])[0]
            confidence = max(proba) * 100
            result = "PHISHING!" if prediction == 1 else "LEGITIMATE"
            print(f"Result: {result} {confidence:.1f}% confidence")
        else:
            result = "PHISHING!" if prediction == 1 else "LEGITIMATE"
            print(f"Result: {result}")
        
        if prediction == 1:
            phishing_count += 1
        else:
            legitimate_count += 1
    
    print(f"\nSummary: {phishing_count} phishing, {legitimate_count} legitimate from {len(emails)} emails")


# Main
def main():
    print("\n-------------------------------------------------------------------")
    print("           ML-BASED EMAIL PHISHING DETECTION SYSTEM")
    print("-------------------------------------------------------------------")    
    
    # Load data
    df = load_data_from_excel()
    
    if df is None:
        print("\nCannot continue without data files")
        return
    
    # Show options
    while True:
        print("\nOptions:")
        print("-------------------------------------------------------------------")
        print("1- Train and compare all models (save the best model)")
        print("2- Load saved model and classify emails")
        print("3- Exit")        
        choice = input("Enter your choice: ").strip()
        
        # Train and evaluate
        if choice == "1":
            while True:
                print("\nOptions:")
                print("-------------------------------------------------------------------")
                print("a- Quick evaluation (run once and save best model)")
                print("b- Full comparison (average over 20 runs)")
                print("c- Back")
                
                mode = input("Enter choice (a, b, or c): ").strip().lower()
                
                if mode == "b":
                    NUM_RUNS = 20
                    all_results = run_comparison(df, NUM_RUNS)
                    best_name, averages = show_results(all_results, NUM_RUNS, len(df))
                    
                    # Now train and save the best model
                    print(f"Saving {best_name} as the best model...")
                    train_and_save_best_model(df)
                elif mode == "a":
                    train_and_save_best_model(df)
                elif mode == "c":
                    break
                else:
                    print("\nError. Choose (a, b, or c)")
        
        # Load and classify
        elif choice == "2":
            model, vectorizer, model_name = load_model_for_classification()
            
            if model is None:
                continue
            
            while True:
                print("\nOptions:")
                print("-------------------------------------------------------------------")
                print("a- Type emails directly")
                print("b- Load emails from a text file")
                print("c- Back")
                
                classify_choice = input("Enter choice (a, b, or c): ").strip().lower()
                
                if classify_choice == "b":
                    print("Emails should be separated by (~)")   

                    filepath = input("\nFile path: ").strip()
                    if filepath:
                        classify_from_file(filepath, model, vectorizer, model_name)
                    else:
                        print("No file path entered.")
                elif classify_choice == "a":
                    print("Emails should be separated by (~)")
                    
                    email_input = input("\nYour emails: ").strip()
                    
                    if email_input:
                        if "~" in email_input:
                            emails = [e.strip() for e in email_input.split("~")]
                        else:
                            emails = [email_input.strip()]
                        emails = [e for e in emails if e]
                        
                        if emails:
                            classify_emails(emails, model, vectorizer, model_name)
                        else:
                            print("No valid emails entered.")
                    else:
                        print("No emails entered.")
                elif classify_choice == "c":
                    break
                else:
                    print("\nError. Choose (a, b, or c)")
        
        elif choice == "3":
            print("\nExiting...")
            break
        
        else:
            print("\nError. Choose (1, 2, or 3)")


if __name__ == "__main__":
    main()
