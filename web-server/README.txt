This file should contain:

  - Timothy Goodwin
  - tlg2132
  - lab 7
 
 part1:
 My website hosts two images and some html text. Note, I have taken an image off of Jae's homepage
and modified it to create a new and original piece of work. If it is felt
that I have violated usage permissions, please email me and I will take the
content in question off of the web.

 part2: http-server

 The http-server responds to client requests in HTTP/1.0. It opens and
 maintains a connection to mdb-lookup-server (Jae's version) and closes
 this connection when http-server terminates. The server only supports
 HTTP/1.0 and HTTP/1.1 protocol and will respond with a 501 Not Implemented status
 code if the client's request is either not supported or improperly
 formatted. This server does not support functionality to check whether a
 requested item is either a regular file or a directory. It's behavior is
 undefined when the request URI is a directory without a '/' at the end.
 After each request, the server will log the client's IP address, request method, the request
 URI, and the corresponding status code and reason phrase.
