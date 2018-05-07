
def print_csv_table(r, rows, cols):
    '''
    r: r[row][col]
    '''
    all_line = []
    all_line.append(',' + ','.join(map(str, cols)))
    def quote_if_comma(p):
        if ',' in p:
            return '"' + p + '"'
        else:
            return p
    for row in rows:
        parts = []
        parts.append(row)
        for col in cols:
            parts.append(r[row][col])
        all_line.append(','.join(map(quote_if_comma, map(str, parts))))
    return '\n'.join(all_line)

def cartesian_index(sizes):
    index = [0] * len(sizes)
    yield index

    def is_end(index, sizes):
        for i in range(len(index)):
            if index[i] != sizes[i] - 1:
                return False
        return True

    while not is_end(index, sizes):
        found = -1
        for i in range(len(index) - 1, -1, -1):
            if index[i] == sizes[i] - 1:
                index[i] = 0
            else:
                found = i
                break
        if found != -1:
            index[found] = index[found] + 1
            yield index

def _spans(sizes):
    span = []
    for i, s in enumerate(sizes):
        x = 1
        for j in xrange(i + 1, len(sizes)):
            x = x * sizes[j]
        span.append(x)
    return span

def _dup(sizes):
    dup = []
    k = 1
    for i in xrange(len(sizes)):
        dup.append(k)
        k = k * sizes[i]
    return dup

def _extract(r, names, index):
    x = r 
    for i in range(len(index)):
        x = x[names[i][index[i]]]
    return x

def print_m_table(r, all_rows, all_cols, caption=None):
    sizes_cols = map(len, all_cols)
    cols_dup = _dup(sizes_cols)
    cols_span = _spans(sizes_cols)
    num_cols = cols_span[0] * sizes_cols[0] + len(all_rows)

    lines = []
    lines.append('\\begin{table}[H]')
    lines.append('\\centering')
    if caption:
        lines.append('\\caption{{{}}}'.format(caption))
        lines.append('\\label{{{}}}'.format(caption.replace(' ', '_')))
    line = '\\begin{{tabular}}{{{}@{{}}}}'.format('@{~}c' * num_cols)
    lines.append(line)
    lines.append('\\toprule')

    for i in xrange(len(all_cols)):
        line = ''
        for j in xrange(len(all_rows) - 1):
            line = line + '&'
        s = cols_span[i]
        for j in range(cols_dup[i]):
            for k in xrange(len(all_cols[i])):
                if s == 1:
                    line = line + '&{}'.format(all_cols[i][k])
                else:
                    line = line + '&\multicolumn{{{0}}}{{c}}{{{1}}}'.format(s,
                            all_cols[i][k])
        line = line + '\\\\'
        lines.append(line)
        lines.append('\\midrule')
    sizes_rows = map(len, all_rows)
    rows_span = _spans(sizes_rows)
    digit_format = '&{}'
    for index in cartesian_index(sizes_rows):
        line = ''
        for i in range(len(index)):
            prefix = '' if i == 0 else '&'
            if all(v == 0 for v in index[i + 1: ]):
                if rows_span[i] == 1:
                    line = '{}{}{}'.format(line, prefix,
                            all_rows[i][index[i]])
                else:
                    line = line + prefix + \
                            '\multirow{{{0}}}{{*}}{{{1}}}'.format(rows_span[i],
                                            all_rows[i][index[i]])
            else:
                if rows_span[i] == 1:
                    line = '{}{}{}'.format(line, prefix, all_rows[i][index[i]])
        for col_index in cartesian_index(sizes_cols):
            value = _extract(_extract(r, all_rows, index), all_cols, col_index)
            line = line + digit_format.format(value)
        line = line + '\\\\'
        lines.append(line)
        is_end_first_index = True 
        for i in xrange(1, len(index)):
            if index[i] != sizes_rows[i] - 1:
                is_end_first_index = False
                break
        if is_end_first_index:
            if index[0] != sizes_rows[0] - 1:
                lines.append('\\midrule')

    lines.append('\\bottomrule')
    lines.append('\\end{tabular}')
    lines.append('\\end{table}')

    return '\n'.join(lines)

def print_table(r, rows, cols):
    return print_m_table(r, [rows], [cols])
    #effective_cols = []
    #for row in rows:
        #if len(effective_cols) == 0:
            #for c in cols:
                #if c in r[row]:
                    #effective_cols.append(c)
        #for c in cols:
            #if c in effective_cols:
                #assert c in r[row]
    #cols = effective_cols

    #lines = []
    #lines.append('\\begin{table}')
    #lines.append('\\centering')
    #lines.append('\\caption{{IoU@{{{}}} -- {}}}'.format(eval_map,
        #label_removed))
    #line = '\\begin{{tabular}}{{{}@{{}}}}'.format('@{~}c' * (len(cols) + 1))
    #lines.append(line)
    #lines.append('\\toprule')
    #lines.append('& {}\\\\'.format(' & '.join(cols)))
    #lines.append('\\midrule')
    #for row in rows: 
        #if row not in r:
            #continue
        #line = '{} & {}\\\\'.format(row, 
                #' & '.join(('{0:.2f}'.format(r[row][l]) for l in cols)))
        #lines.append(line)
    #lines.append('\\bottomrule')
    #line = '\\end{tabular}'
    #lines.append(line)
    #line = '\\end{table}'
    #lines.append(line)
    #return '\n'.join(lines)

def _test_print_m_table():
    r = {}
    r['dog'] = {}
    r['dog']['dog1'] = {}
    r['dog']['dog1']['s'] = {}
    r['dog']['dog1']['s']['s1'] = 0
    r['dog']['dog1']['s']['s2'] = 1
    r['dog']['dog2'] = {}
    r['dog']['dog2']['s'] = {}
    r['dog']['dog2']['s']['s1'] = 2
    r['dog']['dog2']['s']['s2'] = 3

    print print_m_table(r, [['dog'], ['dog1', 'dog2']], [['s'], ['s1',
        's2']])

if __name__ == '__main__':
    _test_print_m_table()
    from qd_common import read_to_buffer
    context = read_to_buffer('/home/jianfw/tmp.json')
    result = json.loads(context)
    print_m_table

