# %% ENV
#import psycopg2
import pyodbc as pod
import pandas as pd
import pandas.io.sql as psql
import json

# %% DESCRIPTION
# data InOut
"""
collection of function to handle csv and database tables in and out
"""

# %% FUNCTIONS


def pg_connect2db():
    # define our connection string
    conn_string = "host='localhost' \
        dbname='mydb' \
        user='postgres' \
        password='secret'"
    # print the connection string we will use to connect
    print("Connecting to database\n%s" % conn_string)
    # get a connection, if a connect cannot be made
    # an exception will be raised here
    conn =1 #= psycopg2.connect(conn_string)
    return conn


def pg_table2df(conn, table):
    return psql.read_sql_query("select * from %s" % table, conn)


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


def odbc_connectToServer(server, database):
    '''
    :param server: ssms server to connect.
    :param database: ssms database to connect.
    Makes a connction to SQL server
    Initial version simple - more parameterise later
    '''
    # load default setting
    default_list = json.load(open('./setting/config.json'))["SQL"]
    default_list["server"]
    if server is None:
        server = default_list["server"]
    if database is None:
        database = default_list["database"]
    # Make connection
    sql_conn = pod.connect(
        "Driver={ODBC Driver 13 for SQL Server}; \
        Server=%s; \
        Database=%s;\
        Trusted_Connection=yes;" % (server, database)
    )
    # Return
    return sql_conn


def odbc_StoreProc2Df(stored_proc, **kwargs):
    '''
    :param stored_proc: store proc to fire
    :param **kwargs: optional args (eg server).
    Initial version simple - parameterise later
    '''
    # optional parameter
    server = kwargs.get('server', None)
    database = kwargs.get('database', None)
    # Make connection
    sql_conn = odbc_connectToServer(server, database)
    # Create the cursor
    sql_cursor = sql_conn.cursor()
    # Run store proc
    sql_cursor.execute('exec %s;' % (stored_proc))
    # Get column names fro the cursor description using a list comprehension
    column_names = [column[0] for column in sql_cursor.description]
    # Collect the results (rows or data as alist)
    sql_results = sql_cursor.fetchall()
    # Close theconnection
    sql_conn.close()
    # Convert list to pandas
    results = pd.DataFrame.from_records(sql_results, columns=column_names)
    # Return
    return results


def odbc_Query2Df(query, **kwargs):
    '''
    :param query: query to execute on tables or views.
     :param **kwargs: optional args (eg server).
    '''
    # optional parameter
    server = kwargs.get('server', None)
    database = kwargs.get('database', None)
    # Make connection
    sql_conn = odbc_connectToServer(server, database)
    # Run query
    sql_results = pd.read_sql(sql=query, con=sql_conn)
    # Close theconnection
    sql_conn.close()
    # Return
    return sql_results


# %% EXAMPLES

"""
# conn to db
db_conn = db_connect2db()
# download table
table = db_table2df(db_conn,"my_table")
# Create the sql string
sql_string = "dpretailer.uspRetrieveDPRetailerActuals
@DomainProductGroupName = '%s', @CountryISOCode = %s" % (
     'AIR CONDITIONER',
    504
)
# Grab the actuals from database
dat = odbc_StoreProc2Df(sql_string)
"""
