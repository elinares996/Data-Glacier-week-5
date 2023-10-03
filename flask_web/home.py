from flask import Flask, render_template, request, jsonify
import pickle
import pandas as pd

app = Flask(__name__)

# Load the model once when the app starts
model = pickle.load(open('model.pkl', 'rb'))


@app.route('/', methods=['GET', 'POST'])
def home():
    prediction = None
    if request.method == 'GET':
        flight_hour = request.form.get('flight')
        length_of_stay = request.form.get('length of stay')
        num_passengers = request.form.get('number of pass')
        purchase_lead = request.form.get('purchase')
        booking_complete = request.form.get('booking')
        wants_in_flight_meals = request.form.get('meals')
        wants_preferred_seat = request.form.get('preffered')

        test_df = pd.DataFrame(
            {
                "num_passengers": [num_passengers],
                "purchase_lead": [purchase_lead],
                "length_of_stay": [length_of_stay],
                "flight_hour": [flight_hour],
                "wants_preferred_seat": [wants_preferred_seat],
                "wants_in_flight_meals": [wants_in_flight_meals],
                "booking_complete": [booking_complete]
            })
        test_df = test_df.fillna(0)
        pred_duration = model.predict(test_df)

        # Assuming model.predict returns an array
        prediction = str(pred_duration)

    if request.method == 'POST':
        flight_hour = request.form.get('flight')
        length_of_stay = request.form.get('length of stay')
        num_passengers = request.form.get('number of pass')
        purchase_lead = request.form.get('purchase')
        booking_complete = request.form.get('booking')
        wants_in_flight_meals = request.form.get('meals')
        wants_preferred_seat = request.form.get('preffered')

        test_df = pd.DataFrame(
            {
                "num_passengers": [num_passengers],
                "purchase_lead": [purchase_lead],
                "length_of_stay": [length_of_stay],
                "flight_hour": [flight_hour],
                "wants_preferred_seat": [wants_preferred_seat],
                "wants_in_flight_meals": [wants_in_flight_meals],
                "booking_complete": [booking_complete]
            })
        test_df = test_df.fillna(0)
        pred_duration = model.predict(test_df)

        # Assuming model.predict returns an array
        prediction = str(pred_duration)

    return render_template('index.html', prediction=prediction)


if __name__ == '__main__':
    app.run(debug=True, use_reloader=False)
