from flask import Flask, request, jsonify, render_template
import pandas as pd
from mlxtend.frequent_patterns import apriori, association_rules
from werkzeug.utils import secure_filename
import os

app = Flask(__name__)

# Uploads folder setup
UPLOAD_FOLDER = 'uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

# Global DataFrame for rules
global_rules = pd.DataFrame()

@app.route('/')
def index():
    return render_template('index.html')  # not used in embedded HTML, but safe to keep

@app.route('/upload', methods=['POST'])
def upload_csv():
    global global_rules

    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400

    try:
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)

        df = pd.read_csv(filepath)

        # Ensure required columns
        if 'Transaction' not in df.columns or 'Item' not in df.columns:
            return jsonify({'error': 'CSV must contain "Transaction" and "Item" columns'}), 400

        # Lowercase items for consistency
        df['Item'] = df['Item'].astype(str).str.strip().str.lower()

        # Create basket matrix (one-hot encoded)
        basket = df.groupby(['Transaction', 'Item'])['Item'].count().unstack().fillna(0)
        basket = basket.applymap(lambda x: 1 if x > 0 else 0)

        # Generate frequent itemsets
        frequent_itemsets = apriori(basket, min_support=0.01, use_colnames=True)

        # Generate rules
        rules = association_rules(frequent_itemsets, metric="lift", min_threshold=1.0)
        global_rules = rules.copy()

        # Convert rules to JSON
        result = []
        for _, row in rules.iterrows():
            result.append({
                'antecedents': list(row['antecedents']),
                'consequents': list(row['consequents']),
                'support': round(row['support'], 4),
                'confidence': round(row['confidence'], 4),
                'lift': round(row['lift'], 4)
            })

        return jsonify(result)

    except Exception as e:
        return jsonify({'error': f"Processing failed: {str(e)}"}), 500


@app.route('/recommend', methods=['POST'])
def recommend_product():
    global global_rules
    try:
        data = request.get_json()
        if not data or 'product' not in data:
            return jsonify({'error': 'Missing product in request body'}), 400

        product = data['product'].strip().lower()
        recommendations = set()

        for _, row in global_rules.iterrows():
            antecedents = set([str(i).lower() for i in row['antecedents']])
            consequents = set([str(i).lower() for i in row['consequents']])
            if product in antecedents:
                recommendations.update(consequents)

        return jsonify({'recommendations': sorted(recommendations)})

    except Exception as e:
        return jsonify({'error': f"Recommendation error: {str(e)}"}), 500


if __name__ == '__main__':
    app.run(debug=True)

