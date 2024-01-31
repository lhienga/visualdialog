from . import db
from flask_login import UserMixin
from sqlalchemy.sql import func


class Chat(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    img_url = db.Column(db.String(10000))
    user_msg = db.Column(db.String(10000))
    bot_msg = db.Column(db.String(10000))
    date = db.Column(db.DateTime(timezone=True), default=func.now())
    feedback = db.Column(db.String(10000))
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'))


class User(db.Model, UserMixin):
    id = db.Column(db.Integer, primary_key=True)
    email = db.Column(db.String(150), unique=True)
    password = db.Column(db.String(150))
    first_name = db.Column(db.String(150))
    chat = db.relationship('Chat')
