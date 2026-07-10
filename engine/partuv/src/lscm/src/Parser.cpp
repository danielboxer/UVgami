#include "Parser.h"

using namespace MeshLib;

Parser::~Parser() {
	ListNode<Token> * node = m_tokens.head();
	while( node != NULL ) {
		delete node->data();
		node->data() = NULL;
		node = node->next();
	}
}

Parser::Parser( std::string & str ) {
	//copy string
	int i;
	for( i = 0; i < (signed)( str.length()); i ++ ) {
		m_line[i] = str.c_str()[i];
	}
	m_line[i] = 0;

	m_pt = m_line;

	while( !end() ) {
		skip_blank();
		if( end() ) break;

		//copy key
		char * pkey = m_key;
		char   ch = next_char();

		while( ch != ' ' && ch != '=' && !end() ) {
			*pkey ++ = ch;
			ch = next_char();
		}
		if( ch != '=' && ch != ' ')
			*pkey++ = ch;

		*pkey = 0;

		if( ch == ' ' ) {
			skip_blank();
		}

		if( ch != '=' ) {
			Token *	tk = new Token;
			
			tk->m_key  = std::string( m_key );
			
			m_tokens.Append( tk );
			continue;
		}

		if( end() ) break;

		ch = next_char();

		while( ch != '(' ) ch = next_char();

		char * pvalue = m_value;

		while( ch != ')' && !end() ) {
			*pvalue ++ = ch;
			ch = next_char();
		}
		*pvalue++ = ch;
		*pvalue  = 0;

		Token *	tk = new Token;
		
		tk->m_key   = std::string( m_key   );
		tk->m_value = std::string( m_value );
		
		m_tokens.Append( tk );
	}
}

char Parser::next_char() {
	char ch= *m_pt;
	m_pt++;
	return ch;
}

void Parser::skip_blank() {
	while( *m_pt == ' ' ) m_pt++;
}

bool Parser::end() {
	return ( (*m_pt) == 0 );
}
