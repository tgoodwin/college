
#web-server

###This project is a hand-made, home cooked HTTP server.


 The http-server responds to client requests in HTTP/1.0. It opens and
 maintains a connection to a database server called 'mdb-lookup-server' and closes
 this connection when http-server terminates. The server only supports
 HTTP/1.0 and HTTP/1.1 protocol and will respond with a 501 Not Implemented status
 code if the client's request is either not supported or improperly
 formatted. This server does not support functionality to check whether a
 requested item is either a regular file or a directory. It's behavior is
 undefined when the request URI is a directory without a '/' at the end.
 After each request, the server will log the client's IP address, request method, the request
 URI, as well as the corresponding status code and reason phrase.
