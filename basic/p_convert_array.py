import numpy as np
from mltools import mdaio
import os

DEBUG=True

def file_extension(fname):
    filename, ext = os.path.splitext(fname)
    return ext

def determine_file_format(ext,dimensions):
    if ext=='.mda':
        return 'mda'
    elif ext=='.npy':
        return 'npy'
    else:
        return 'dat'

def npy_dtype_to_string(dt):
    str=dt.str[1:]
    map={
        "f2":'float16',
        "f4":'float32',
        "f8":'float64',
        "i1":'int8',
        "i2":'int16',
        "i4":'int32',
        "u2":'uint16',
        "u4":'uint32'
    }
    return map[str]

def get_num_bytes_per_entry_from_dt(dt):
    if dt == 'uint8':
        return 1
    if dt == 'float32':
        return 4
    if dt == 'int16':
        return 2
    if dt == 'int32':
        return 4
    if dt == 'uint16':
        return 2
    if dt == 'float64':
        return 8
    if dt == 'uint32':
        return 4
    return None

def copy_raw_file_data(input,output,*,start_byte,num_entries,dtype,dtype_out):
    if dtype != dtype_out:
        raise Exception('Copying from one datatype to another not yet implemented')
    size1=get_num_bytes_per_entry_from_dt(dtype)
    size2=get_num_bytes_per_entry_from_dt(dtype_out)
    bufsize=10000
    with open(input,'rb') as f1:
        f1.seek(start_byte)
        with open(output,'wb') as f2:
            while num_entries:
                chunk_num_entries = min(bufsize,num_entries)
                data = f1.read(chunk_num_entries*size1)
                ## convert data here
                f2.write(data)
                num_entries -= chunk_num_entries

def determine_npy_header_size(fname):
    with open(fname,'rb') as f:
        header_str=f.readline() ## I think it ends with \n
        return f.tell()

processor_name='ephys.convert_array'
processor_version='0.1'
def convert_array(*,input,output,format='',format_out='',dimensions='',dtype='',dtype_out=''):
    """
    Convert a multi-dimensional array between various formats ('.mda', '.npy', '.dat') based on the file extensions of the input/output files

    Parameters
    ----------
    input : INPUT
        Path of input array file.
    output : OUTPUT
        Path of the output array file.
        
    format : string
        The format for the input array (mda, npy, dat), or determined from the file extension if empty
    format_out : string
        The format for the output input array (mda, npy, dat), or determined from the file extension if empty
    dimensions : string
        Comma-separated list of dimensions (shape). If empty, it is auto-determined, if possible, by the input array.
    dtype : string
        The data format for the input array. Choices: int8, int16, int32, uint16, uint32, float32, float64 (possibly float16 in the future).
    dtype_out : string
        The data format for the output array. If empty, the dtype for the input array is used.
        
    """    
    format_in=format
    if not format_in:
        format_in=determine_file_format(file_extension(input),dimensions)
    if not format_out:
        format_out=determine_file_format(file_extension(output),dimensions)
    print ('Input/output formats: {}/{}'.format(format_in,format_out))

    dims=None

    if (format_in=='mda') and (dtype==''):
        header=mdaio.readmda_header(input)
        dtype=header.dt
        dims=header.dims

    if (format_in=='npy') and (dtype==''):
        A=np.load(input,mmap_mode='r')
        dtype=npy_dtype_to_string(A.dtype)
        dims=A.shape
        A=0

    if dimensions:
        dims2=[int(entry) for entry in dimensions.split(',')]
        if dims:
            if len(dims) != len(dims2):
                raise Exception('Inconsistent number of dimensions for input array')
            if not np.all(np.array(dims)==np.array(dims2)):
                raise Exception('Inconsistent dimensions for input array')
        dims=dims2

    if not dtype_out:
        dtype_out=dtype

    if not dtype:
        raise Exception('Unable to determine datatype for input array')

    if not dtype_out:
        raise Exception('Unable to determine datatype for output array')
    
    if (dims[1] == -1) and (dims[0] > 0):
        if (dtype) and (format_in=='dat'):
            bits      = int(dtype[-2:]) # number of bits per entry of dtype
            filebytes = os.stat(input).st_size # bytes in input file
            entries   = int(filebytes/(int(bits/8))) # entries in input file
            dims[1]   = int(entries/dims[0]) # caclulated second dimension
            if DEBUG:
                print(bits)
                print(filebytes)
                print(int(filebytes/(int(bits/8))))
                print(dims)

    if not dims:       
        raise Exception('Unable to determine dimensions for input array')

    print ('Using dtype={}, dtype_out={}, dimensions={}'.format(dtype,dtype_out,','.join(str(item) for item in dims)))

    if (format_in==format_out) and ((dtype==dtype_out) or (dtype_out=='')):
        print ('Simply copying file...')
        shutil.copyfile(input,output)
        print ('Done.')
        return True

    if format_out=='dat':
        if format_in=='mda':
            H=mdaio.readmda_header(input)
            copy_raw_file_data(input,output,start_byte=H.header_size,num_entries=np.product(dims),dtype=dtype,dtype_out=dtype_out)
            return True
        elif format_in=='npy':
            print ('Warning: loading entire array into memory. This should be avoided in the future.')
            A=np.load(input,mmap_mode='r').astype(dtype=dtype_out,order='F',copy=False)
            A=A.ravel(order='F')
            A.tofile(output)
            # The following was problematic because of row-major ordering, i think
            #header_size=determine_npy_header_size(input)
            #copy_raw_file_data(input,output,start_byte=header_size,num_entries=np.product(dims),dtype=dtype,dtype_out=dtype_out)
            return True
        elif format_in=='dat':
            raise Exception('This case not yet implemented.')
        else:
            raise Exception('Unexpected case.')

    elif (format_out=='mda') or (format_out=='npy'):
        if format_in=='npy':
            print ('Warning: loading entire array into memory. This should be avoided in the future.')
            A=np.load(input,mmap_mode='r').astype(dtype=dtype,order='F',copy=False)
            if format_out=='mda':
                mdaio.writemda(A,output,dtype=dtype_out)
            else:
                mdaio.writenpy(A,output,dtype=dtype_out)
            return True
        elif format_in=='dat':
            print ('Warning: loading entire array into memory. This should be avoided in the future.')
            A=np.fromfile(input,dtype=dtype,count=np.product(dims));
            A=A.reshape(tuple(dims),order='F')
            if format_out=='mda':
                mdaio.writemda(A,output,dtype=dtype_out)
            else:
                mdaio.writenpy(A,output,dtype=dtype_out)
            return True
        elif format_in=='mda':
            print ('Warning: loading entire array into memory. This should be avoided in the future.')
            A=mdaio.readmda(input)
            if format_out=='mda':
                mdaio.writemda(A,output,dtype=dtype_out)
            else:
                mdaio.writenpy(A,output,dtype=dtype_out)
            return True
        else:
            raise Exception('Unexpected case.')
    else:
        raise Exception('Unexpected output format: {}'.format(format_out))

    raise Exception('Unexpected error.')

convert_array.name=processor_name
convert_array.version=processor_version

def test_convert_array(dtype='int32',shape=[12,3,7]):
    X=np.array(np.random.normal(0,1,shape),dtype=dtype)
    np.save('test_convert1.npy',X)
    convert_array(input='test_convert1.npy',output='test_convert2.mda') # npy -> mda
    convert_array(input='test_convert2.mda',output='test_convert3.npy') # mda -> npy
    convert_array(input='test_convert3.npy',output='test_convert4.dat') # npy -> dat
    convert_array(input='test_convert4.dat',output='test_convert5.npy',dtype=dtype,dimensions=','.join(str(entry) for entry in X.shape))  # dat -> npy
    convert_array(input='test_convert5.npy',output='test_convert6.mda') # npy -> mda
    convert_array(input='test_convert6.mda',output='test_convert7.dat') # mda -> dat
    convert_array(input='test_convert7.dat',output='test_convert8.mda',dtype=dtype,dimensions=','.join(str(entry) for entry in X.shape)) # dat -> mda
    Y=mdaio.readmda('test_convert8.mda')
    print(np.max(np.abs(X-Y)),Y.dtype)
