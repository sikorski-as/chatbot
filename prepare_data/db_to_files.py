import sqlite3
import pandas as pd

if __name__ == '__main__':

    timeframes = ['2015-07']

    for timeframe in timeframes:
        connection = sqlite3.connect('{}.db'.format(timeframe))
        c = connection.cursor()
        limit = 10000
        last_unix = 0
        cur_length = limit
        counter = 0
        test_done = False

        while cur_length == limit and counter != 10:

            df = pd.read_sql(
                f"SELECT * FROM parent_reply WHERE unix > {last_unix} and parent NOT NULL and score > 0 ORDER BY unix ASC LIMIT {limit}",
                connection
            )
            last_unix = df.tail(1)['unix'].values[0]
            cur_length = len(df)

            if not test_done:
                # with open('output_files/small.from', 'a', encoding='utf8') as f:
                #     for content in df['parent'].values:
                #         f.write(content + '\n')
                #
                # with open('output_files/small.to', 'a', encoding='utf8') as f:
                #     for content in df['comment'].values:
                #         f.write(str(content) + '\n')

                test_done = True

            else:
                with open('output_files/train3.from', 'a', encoding='utf8') as f:
                    for content in df['parent'].values:
                        f.write(content + '\n')

                with open('output_files/train3.to', 'a', encoding='utf8') as f:
                    for content in df['comment'].values:
                        f.write(str(content) + '\n')

            counter += 1
            if counter % 20 == 0:
                print(counter * limit, 'rows completed so far')
