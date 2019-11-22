# Chapter 3. Classification

## MNIST

Often considered the "Hello World" of ML, Scikit-Learn offers helper functions for downloading the MNIST 


```
from sklearn.datasets import fetch_mldata
mnist = fetch_mldata('MNIST original', data_home='assets/scikit_learn_data/')
mnist
```


    ---------------------------------------------------------------------------

    TimeoutError                              Traceback (most recent call last)

    /opt/anaconda3/envs/daysOfCode-env/lib/python3.7/urllib/request.py in do_open(self, http_class, req, **http_conn_args)
       1316                 h.request(req.get_method(), req.selector, req.data, headers,
    -> 1317                           encode_chunked=req.has_header('Transfer-encoding'))
       1318             except OSError as err: # timeout error


    /opt/anaconda3/envs/daysOfCode-env/lib/python3.7/http/client.py in request(self, method, url, body, headers, encode_chunked)
       1243         """Send a complete request to the server."""
    -> 1244         self._send_request(method, url, body, headers, encode_chunked)
       1245 


    /opt/anaconda3/envs/daysOfCode-env/lib/python3.7/http/client.py in _send_request(self, method, url, body, headers, encode_chunked)
       1289             body = _encode(body, 'body')
    -> 1290         self.endheaders(body, encode_chunked=encode_chunked)
       1291 


    /opt/anaconda3/envs/daysOfCode-env/lib/python3.7/http/client.py in endheaders(self, message_body, encode_chunked)
       1238             raise CannotSendHeader()
    -> 1239         self._send_output(message_body, encode_chunked=encode_chunked)
       1240 


    /opt/anaconda3/envs/daysOfCode-env/lib/python3.7/http/client.py in _send_output(self, message_body, encode_chunked)
       1025         del self._buffer[:]
    -> 1026         self.send(msg)
       1027 


    /opt/anaconda3/envs/daysOfCode-env/lib/python3.7/http/client.py in send(self, data)
        965             if self.auto_open:
    --> 966                 self.connect()
        967             else:


    /opt/anaconda3/envs/daysOfCode-env/lib/python3.7/http/client.py in connect(self)
        937         self.sock = self._create_connection(
    --> 938             (self.host,self.port), self.timeout, self.source_address)
        939         self.sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)


    /opt/anaconda3/envs/daysOfCode-env/lib/python3.7/socket.py in create_connection(address, timeout, source_address)
        726     if err is not None:
    --> 727         raise err
        728     else:


    /opt/anaconda3/envs/daysOfCode-env/lib/python3.7/socket.py in create_connection(address, timeout, source_address)
        715                 sock.bind(source_address)
    --> 716             sock.connect(sa)
        717             # Break explicitly a reference cycle


    TimeoutError: [Errno 60] Operation timed out

    
    During handling of the above exception, another exception occurred:


    URLError                                  Traceback (most recent call last)

    <ipython-input-3-7d01ea864b02> in <module>
          1 from sklearn.datasets import fetch_mldata
    ----> 2 mnist = fetch_mldata('MNIST original', data_home='assets/scikit_learn_data/')
          3 mnist


    /opt/anaconda3/envs/daysOfCode-env/lib/python3.7/site-packages/sklearn/utils/deprecation.py in wrapped(*args, **kwargs)
         84         def wrapped(*args, **kwargs):
         85             warnings.warn(msg, category=DeprecationWarning)
    ---> 86             return fun(*args, **kwargs)
         87 
         88         wrapped.__doc__ = self._update_doc(wrapped.__doc__)


    /opt/anaconda3/envs/daysOfCode-env/lib/python3.7/site-packages/sklearn/datasets/mldata.py in fetch_mldata(dataname, target_name, data_name, transpose_data, data_home)
        124         urlname = MLDATA_BASE_URL % quote(dataname)
        125         try:
    --> 126             mldata_url = urlopen(urlname)
        127         except HTTPError as e:
        128             if e.code == 404:


    /opt/anaconda3/envs/daysOfCode-env/lib/python3.7/urllib/request.py in urlopen(url, data, timeout, cafile, capath, cadefault, context)
        220     else:
        221         opener = _opener
    --> 222     return opener.open(url, data, timeout)
        223 
        224 def install_opener(opener):


    /opt/anaconda3/envs/daysOfCode-env/lib/python3.7/urllib/request.py in open(self, fullurl, data, timeout)
        523             req = meth(req)
        524 
    --> 525         response = self._open(req, data)
        526 
        527         # post-process response


    /opt/anaconda3/envs/daysOfCode-env/lib/python3.7/urllib/request.py in _open(self, req, data)
        541         protocol = req.type
        542         result = self._call_chain(self.handle_open, protocol, protocol +
    --> 543                                   '_open', req)
        544         if result:
        545             return result


    /opt/anaconda3/envs/daysOfCode-env/lib/python3.7/urllib/request.py in _call_chain(self, chain, kind, meth_name, *args)
        501         for handler in handlers:
        502             func = getattr(handler, meth_name)
    --> 503             result = func(*args)
        504             if result is not None:
        505                 return result


    /opt/anaconda3/envs/daysOfCode-env/lib/python3.7/urllib/request.py in http_open(self, req)
       1343 
       1344     def http_open(self, req):
    -> 1345         return self.do_open(http.client.HTTPConnection, req)
       1346 
       1347     http_request = AbstractHTTPHandler.do_request_


    /opt/anaconda3/envs/daysOfCode-env/lib/python3.7/urllib/request.py in do_open(self, http_class, req, **http_conn_args)
       1317                           encode_chunked=req.has_header('Transfer-encoding'))
       1318             except OSError as err: # timeout error
    -> 1319                 raise URLError(err)
       1320             r = h.getresponse()
       1321         except:


    URLError: <urlopen error [Errno 60] Operation timed out>



```

```
