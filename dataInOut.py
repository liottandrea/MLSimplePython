# %% ENV
import psycopg2
import pandas as pd
import pandas.io.sql as psql

# %% DESCRIPTION
# data InOut
"""
collection of function to handle csv and database tables in and out
"""

# %% FUNCTIONS


# Connect to the db
def db_connect2db():
    # define our connection string
    conn_string = "host='localhost' dbname='mydb' user='postgres' password='secret'"
    # print the connection string we will use to connect
    print("Connecting to database\n%s" % conn_string)
    # get a connection, if a connect cannot be made
    # an exception will be raised here
    conn = psycopg2.connect(conn_string)
    return conn

# select a table


def db_table2df(conn, table):
    return psql.read_sql_query("select * from %s" % table, conn)


# functions
def csv_df2Xy(csv_file, x_cols, y_col):
    # read csv
    dataset = pd.read_csv(csv_file)
    # divide x and y
    X = dataset[x_cols]
    y = dataset[y_col]
    # force to dataframe in case only one column
    if isinstance(X, pd.Series):
        X = X.to_frame()
    if isinstance(y, pd.Series):
        y = y.to_frame()
    # return
    return X, y

# %% EXAMPLES


"""
# conn to db
db_conn = db_connect2db()
# download table
table = db_table2df(db_conn,"my_table")
"""
