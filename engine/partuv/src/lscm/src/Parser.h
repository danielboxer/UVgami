#ifndef _MESHLIB_PARSER_H_
#define _MESHLIB_PARSER_H_

#include <string>

namespace MeshLib {
class Token {
public:
	std::string m_key;
	std::string m_value;
};

class Parser {
public:
	Parser( std::string & str);
	~Parser();

	List<Token> & tokens() { return m_tokens; };

private:
	char next_char();
	void skip_blank();
	bool end();

	List<Token> m_tokens;

	char m_line[1024];
	char m_key[1024];
	char m_value[1024];

	char * m_pt;
};
}

#endif
