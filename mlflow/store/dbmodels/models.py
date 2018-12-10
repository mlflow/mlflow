import sqlalchemy
from sqlalchemy.ext.declarative import declarative_base

Base = declarative_base()


class Experiment(Base):
    __tablename__ = 'experiments'
    id = sqlalchemy.Column(sqlalchemy.Integer, primary_key=True)
    name = sqlalchemy.Column(sqlalchemy.TEXT, unique=True)

    def __repr__(self):
        return '<Experiment {} - {}>'.format(self.name, self.id)
