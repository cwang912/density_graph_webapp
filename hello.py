from flask import Flask, render_template, make_response, send_file, redirect, url_for
import numpy as np
import cStringIO
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sb
import geocoder
from math import radians, cos, sin, asin, sqrt
from flask.ext.script import Manager
from flask.ext.bootstrap import Bootstrap
from flask.ext.moment import Moment
from flask.ext.wtf import Form
from wtforms import StringField, SubmitField
from wtforms.validators import Required

app = Flask(__name__)
app.config['SECRET_KEY'] = 'hard to guess string'

manager = Manager(app)
bootstrap = Bootstrap(app)
moment = Moment(app)

# HELPER FUNCTIONS:
def haversine(lon1, lat1, lon2, lat2):
    """
    Calculate the great circle distance between two points
    on the earth (specified in decimal degrees)
    """
    # convert decimal degrees to radians
    lon1, lat1, lon2, lat2 = map(radians, [lon1, lat1, lon2, lat2])

    # haversine formula
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
    c = 2 * asin(sqrt(a))
    r = 3956 # 6371 radius of earth in kilometers. Use 3956 for miles
    return c * r

def plot(lat, lon, df, place, dist):
    plt.figure(figsize=(8,6))
    df['dist'] = df.apply(lambda x: haversine(lat, lon, x['lat'], x['lon']), axis=1)
    df = df[df['dist'] < dist].reset_index(drop=True)
    regplot = sb.regplot('dist','density', df, ci=95, lowess=True, truncate=True, \
               scatter_kws={"color": 'slategray', 'alpha':0.8}, line_kws={"color": 'red', "alpha":0.8})

    plt.xlim(xmin=-1, xmax=21)
    plt.xlabel("Distance (miles)")
    plt.ylabel("Density (people per square mile)")
    plt.title(place)
    return regplot

def plot_fun(string, df):
    g = geocoder.google(string)
    return plot(g.lat, g.lng, df, g.address, 20)

def make_plot(place):
    df = pd.read_csv('data/tracts.csv')
    line = plot_fun(place, df)
    # Generate the plot
    f = cStringIO.StringIO()
    plt.savefig(f, format='png')
    # Serve up the data
    f.seek(0)
    # data = f.read()
    return send_file(f, mimetype='image/png')

class NameForm(Form):
    name = StringField('Pick a place in the US:', validators=[Required()])
    submit = SubmitField('Submit')

@app.route('/', methods=['GET', 'POST'])
def index():
    name = None
    form = NameForm()
    if form.validate_on_submit():
        name = form.name.data
        print name
        print url_for('graph', place=name)
        return redirect(url_for('graph', place=name))
        # form.name.data = ''
    return render_template('index.html', form=form, name=name)

# FLASK FUNCTIONS
@app.route('/graph/<place>', methods=['GET', 'POST'])
def graph(place=None):
    name = None
    form = NameForm()
    if form.validate_on_submit():
        name = form.name.data
        return redirect(url_for('graph', place=name))
    return render_template("index.html", form=form, name=place)

@app.route('/fig/<place>')
def fig(place):
    df = pd.read_csv('data/tracts.csv')
    line = plot_fun(place, df)
    # Generate the plot
    f = cStringIO.StringIO()
    plt.savefig(f, format='png')
    # Serve up the data
    f.seek(0)
    # data = f.read()
    return send_file(f, mimetype='image/png')


if __name__ == '__main__':
  app.run(debug=True)
