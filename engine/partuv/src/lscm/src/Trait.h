#ifndef _TRAIT_H_
#define _TRAIT_H_
#include <string>
#include <iterator>

namespace MeshLib{

struct string_token_iterator
	: public std::iterator<std::input_iterator_tag, std::string>
{
public:
	string_token_iterator() : str(0), start(0), end(0) {}
	string_token_iterator(const std::string & str_, const char * separator_ = " ") :
		separator(separator_),
		str(&str_),
		end(0)
	{
		find_next();
	}
	string_token_iterator(const string_token_iterator & rhs) :
		separator(rhs.separator),
		str(rhs.str),
		start(rhs.start),
		end(rhs.end)
	{
	}

	string_token_iterator & operator++() {
		find_next();
		return *this;
	}

	string_token_iterator operator++(int) {
		string_token_iterator temp(*this);
		++(*this);
		return temp;
	}

	std::string operator*() const {
		return std::string(*str, start, end - start);
	}

	bool operator==(const string_token_iterator & rhs) const {
		return (rhs.str == str && rhs.start == start && rhs.end == end);
	}

	bool operator!=(const string_token_iterator & rhs) const {
		return !(rhs == *this);
	}

private:

	void find_next(void) {
		start = str->find_first_not_of(separator, end);
		if (start == std::string::npos) {
			start = end = 0;
			str = 0;
			return;
		}

		end = str->find_first_of(separator, start);
	}

	const char * separator;
	const std::string * str;
	std::string::size_type start;
	std::string::size_type end;
};

class Trait {
public:
	Trait(){};
	~Trait(){};
};

}

#endif
