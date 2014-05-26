#include <opencv2/opencv.hpp>
#include <sstream>
#include <map>
#include <SDL.h>
#include <Windows.h>
#include <SDL_ttf.h>
#include <iostream>


const char* fonts[] =
{
	"fonts\\times.ttf",
	"fonts\\arial.ttf",
	"fonts\\georgia.ttf"
};


cv::Mat leastSquares(cv::Mat X, cv::Mat Y)
{
	cv::Mat XTrans;
	cv::transpose(X, XTrans);
	//std::cout << "XT: " << XTrans << "\n";
	cv::Mat XTX = XTrans*X;
	cv::Mat XTXInv = XTX.inv();
	//std::cout << "XTXInv: " << XTXInv << "\n";
	cv::Mat XTXInvX = XTXInv*XTrans;
	cv::Mat coefficients = XTXInvX*Y;
	return coefficients;
}


cv::Mat fitQuadraticToContour(std::vector<cv::Point> contour)
{
	cv::Mat Y(contour.size(), 1, CV_32FC1);
	cv::Mat X(contour.size(), 3, CV_32FC1);
	int degrees = 2;
	for (int i = 0; i < contour.size(); i++)
	{
		//yes, <= degrees, because we want to include power of zero (always 1) up to and including the last term
		for (int j = 0; j <= degrees; j++)
		{
			X.at<float>(i, j) = pow(contour[i].x, j);
		}
		Y.at<float>(i, 0) = contour[i].y;
	}

	//we now have our mapping... So calculate least squares using the normal equation
	return leastSquares(X, Y);

}

//mat Y represents the target dimension of the ground truths
cv::Mat fitDimensionPoly(std::vector<cv::Point> points, cv::Mat Y)
{
	cv::Mat X(points.size(), 6, CV_32FC1);
	for (int i = 0; i < points.size(); i++)
	{
		int c = 0;
		float x = points[i].x;
		float y = points[i].y;
		X.at<float>(i, c++) = 1;
		X.at<float>(i, c++) = x;
		X.at<float>(i, c++) = y;
		X.at<float>(i, c++) = x*x;
		X.at<float>(i, c++) = y*y;
		X.at<float>(i, c++) = x*y;
	}

	return leastSquares(X, Y);
}

/**
	This function makes ground truths by taking a contour and the quadratic curve which fits it
	Then it fits the curve to a pair of fourth (todo, currently SECOND degree!) degree polynomials which represent the correction function...
*/
void fitTheFuckingThing(std::vector<cv::Point> contour, cv::Mat qCoeffs, cv::Mat &XCoeffs, cv::Mat &YCoeffs)
{
	//first classify the points as either upper or lower
	std::vector<cv::Point> upperPoints;
	std::vector<cv::Point> lowerPoints;

	float xmax = FLT_MIN;
	float xmin = FLT_MAX;
	for (int i = 0; i < contour.size(); i++)
	{
		float x = contour[i].x;
		float a = qCoeffs.at<float>(0, 0);
		float b = qCoeffs.at<float>(1, 0);
		float c = qCoeffs.at<float>(2, 0);

		float boundY = a + b*x + c*x*x;

		if (contour[i].y > boundY)
		{
			upperPoints.push_back(contour[i]);
		}

		if (contour[i].y < boundY)
		{
			lowerPoints.push_back(contour[i]);
		}
		//find max and min
		if (x > xmax)
		{
			xmax = x;
		}
		else if (x < xmin)
		{
			xmin = x;
		}
	}


	std::vector<cv::Point> upperGroundTruths;
	std::vector<cv::Point> lowerGroundTruths;

	//get centre Y by fitting ellipse and getting its centre 
	cv::RotatedRect ellipse = cv::fitEllipse(contour);
	cv::Point centrePoint = ellipse.center;

	float lineheight = 30;
	float hlineheight = lineheight / 30;

	float upperXSpacing = upperPoints.size() / (xmax - xmin);
	float lowerXSpacing = lowerPoints.size() / (xmax - xmin);
	//generate lower ground truths
	float startX = xmin;
	for (int i = 0; i < lowerPoints.size(); i++)
	{
		cv::Point newPoint(startX + i * lowerXSpacing, centrePoint.y - hlineheight);
		lowerGroundTruths.push_back(newPoint);
	}
	for (int i = 0; i < upperPoints.size(); i++)
	{
		cv::Point newPoint(startX + i * upperXSpacing, centrePoint.y + hlineheight);
		upperGroundTruths.push_back(newPoint);
	}


	std::vector<cv::Point> assocPoints;
	std::vector<cv::Point> assocGroundTruths;

	//sort everrrything by their X components. Lower first and then upper points.
	auto comparator = [](cv::Point &a, cv::Point &b) -> bool
	{
		return a.x < b.x;
	};

	std::sort(lowerPoints.begin(), lowerPoints.end(), comparator);
	std::sort(upperPoints.begin(), upperPoints.end(), comparator);
	//create points array
	assocPoints.insert(assocPoints.end(), lowerPoints.begin(), lowerPoints.end());
	assocPoints.insert(assocPoints.end(), upperPoints.begin(), upperPoints.end());

	std::sort(lowerGroundTruths.begin(), lowerGroundTruths.end(), comparator);
	std::sort(upperGroundTruths.begin(), upperGroundTruths.end(), comparator);

	assocGroundTruths.insert(assocGroundTruths.end(), lowerGroundTruths.begin(), lowerGroundTruths.end());
	assocGroundTruths.insert(assocGroundTruths.end(), upperGroundTruths.begin(), upperGroundTruths.end());

	//we now have our associated ground truths so... least squaaaareeeeeezzzzzzzzz 8DDDDDDDDDDD
	//basically root of all CV research
	//u must be least squarz cause u is well fit
	//but wait
	//ofuck

	//how can return

	//OUTPUT ARGZZZzzZzz0rz

	cv::Mat XTruths(assocGroundTruths.size(), 1, CV_32FC1);
	cv::Mat YTruths(assocGroundTruths.size(), 1, CV_32FC1);

	for (int i = 0; i < assocGroundTruths.size(); i++)
	{
		XTruths.at<float>(i, 0) = assocGroundTruths[i].x;
		YTruths.at<float>(i, 0) = assocGroundTruths[i].y;
	}

	XCoeffs = fitDimensionPoly(assocPoints, XTruths);
	YCoeffs = fitDimensionPoly(assocPoints, YTruths);

}

//takes a line mask image and returns the coefficients for a least squares approximation of a fourth degree polynomial for the text
void fitLineLine(std::vector<cv::Point> lineContour, cv::Mat &XCoeffs, cv::Mat &YCoeffs)
{
	//we will use the horizontal line which intersects the centre of the ellipse fit to this as the expected set of points
	//that line will be of the average height of the contour that we are currently interested in 
	cv::Rect rect = cv::boundingRect(lineContour);
	cv::Mat filledLine(rect.width, rect.height, CV_8UC1); //this will be a filled version of the contour
	std::vector<std::vector<cv::Point>> contourWrapper;
	contourWrapper.push_back(lineContour);
	cv::drawContours(filledLine, contourWrapper, 0, cv::Scalar(255), CV_FILLED);

	//fit an ellipse to the contour.

	cv::RotatedRect ellipse = cv::fitEllipse(lineContour);

	//make vectors of the X and of the Y coordinates of the points that we are interested in
	cv::Mat X(lineContour.size(), 1, CV_32FC1);
	cv::Mat Y(lineContour.size(), 1, CV_32FC1);

	for (int i = 0; i < lineContour.size(); i++)
	{
		X.at<float>(i, 0) = lineContour[i].x;
		Y.at<float>(i, 0) = lineContour[i].y;
	}

	cv::Mat qCoeffs = fitQuadraticToContour(lineContour);
	fitTheFuckingThing(lineContour, qCoeffs, XCoeffs, YCoeffs);





}


//dat return type. get lists of contours which represent which line they appear on
std::vector<std::vector<std::vector<cv::Point>>> getCharacterContoursByLine(std::vector<std::vector<cv::Point>> const &lines, std::vector <std::vector<cv::Point>> const &characterContours)
{

	std::vector<std::vector<std::vector<cv::Point>>> lineAllocatedContours(lines.size());

	//lines by characters. completely arbitrary decision

	//liit = line iterator

	//zomgerd so many nested fors...
	int linecounter = 0;
	for (auto liit = lines.begin(); liit != lines.end(); liit++, linecounter++)
	{
		//chit = character iterator
		for (auto chit = characterContours.begin(); chit != characterContours.end(); chit++)
		{
			//lpit = line point iterator
			for (auto cpit = chit->begin(); cpit != chit->end(); cpit++)
			{
				//the idea is that if any point in the character intersects line contour, then it should be a part of that line
				bool inpoint = cv::pointPolygonTest(*liit, *cpit, false);

				if (inpoint)
				{

					//if it turns out that this contour is within this line, then we should add it to the vector which represents that line
					lineAllocatedContours[linecounter].push_back(*chit);
					break;
				}

			}
		}
	}

	return lineAllocatedContours;
}

//assume that the image has already been inverted (since this will have been done for the line calculation already)
//function also assumes that at this point we have cropped the image/done necessary processing to make sure we don't detect erroneous contours
std::vector<std::vector<cv::Point>> getCharacterContours(cv::Mat image)
{
	//also might be worth making contours a ref return parameter since I'm not sured if the memory is about to  be copied from contours on return (which will be costly!)
	//might do some masking stuff in here to attempt to get better contour hierarchies (like, have the dot of an i associated with the rest of the letter would be nice)
	
	std::vector<std::vector<cv::Point>> contours;
	std::vector<cv::Vec4i> hierarchy;
	cv::findContours(image, contours, hierarchy, CV_RETR_LIST, CV_CHAIN_APPROX_SIMPLE, cv::Point(0, 0));
	return contours; 
}

std::vector<std::vector<std::vector<cv::Point>>> getContourWords(std::vector<std::vector<std::vector<cv::Point>>> contourLines)
{
	//for each line...
	std::vector<std::vector<std::vector<cv::Point>>> allWords;
	for (auto line = contourLines.begin(); line != contourLines.end(); line++)
	{
		//first we want to get the average position of each of the characters;
		//we will do two iterations of the characters - one to calculate the standard deviation, the second to segment the words
		std::vector<float> xpositions;
		for (auto chr = line->begin(); chr != line->end(); chr++)
		{
			xpositions.push_back(cv::boundingRect(*chr).tl().x);
		}

		std::vector<float> stddevvec;
		std::vector<float> meanvec;
		cv::meanStdDev(xpositions, stddevvec, meanvec);
		float mean = meanvec[0];
		float stddev = stddevvec[0];
		std::vector<std::vector<cv::Point>> currentWord;
		std::vector<std::vector<std::vector<cv::Point>>> wordsForLine;
		float clast = 0;
		for (auto chr = line->begin(); chr != line->end(); chr++)
		{
			float cx = cv::boundingRect(*chr).tl().x;
			//if the current gap is statistically unlikely, consider it a space (hence a new word)
			if (cx - clast > mean + stddev * 2)
			{
				//push the current word onto the list (provided it is not empty):
				if (currentWord.size() != 0)
				{
					wordsForLine.push_back(currentWord);

				}
				currentWord = std::vector<std::vector<cv::Point>>(); //make the next word
			}

			currentWord.push_back(*chr);
		}

		//now we have a list of words for a particular line.
		//allWords.insert(allWords.back(), wordsForLine.begin(), wordsForLine.end());
		for (int i = 0; i < wordsForLine.size(); i++)
		{
			allWords.push_back(wordsForLine[i]);
		}
	}
	return allWords;
}

cv::Mat processBookScan(cv::Mat book)
{
	
	
	//cv::Mat blurredBook(book);
	
	//horizontal blur pass, generate masks

	//invert book
	book = cv::Mat::ones(book.rows, book.cols, book.type()) * 255 - book;


	//character mat at this point is just the inverted original book. We want a copy because the character contours is about to make a mess of it
	


	//make dat blur iterative
	cv::threshold(book, book, 0, 255, CV_THRESH_OTSU | CV_THRESH_BINARY);

	cv::Mat characterMat;
	book.copyTo(characterMat);


	std::vector<std::vector<cv::Point>> charcontours = getCharacterContours(characterMat);
	//oh this is exciting. At this point we have a collection of contours organised by line! :D
	//so let's get a collection of contours ordered by WORD!


	for (int i = 0; i < 20; ++i)
	{
		cv::GaussianBlur(book, book, cv::Size(21, 1), 40, 1);
		//cv::threshold(book, book, 0, 255, CV_THRESH_OTSU | CV_THRESH_BINARY);
	}
	
	//do a single square pass to try to fill in the gaps and ensure that the mask heights are likely to properly encompass our text
	cv::GaussianBlur(book, book, cv::Size(9, 9), 3, 3);
	


	//note: might need to invert before the threshold
	
	cv::Mat thresholdMasks;
	cv::threshold(book, book, 0, 255, CV_THRESH_OTSU | CV_THRESH_BINARY);


	//invert the book

	


	//now find the contours which represent our masks:
	std::vector<std::vector<cv::Point>> contours;
	std::vector<cv::Vec4i> hierarchy;
	cv::findContours(book, contours, hierarchy, CV_RETR_LIST, CV_CHAIN_APPROX_SIMPLE);
	for (int i = 0; i < contours.size(); i++)
	{
		auto ellipse = cv::fitEllipse(contours[i]);
		std::stringstream anglestream;
		anglestream << "Angle: " << ellipse.angle;
		cv::putText(book, anglestream.str(), ellipse.center, CV_FONT_HERSHEY_PLAIN, 1, cv::Scalar(255));
		cv::drawContours(book, contours, i, cv::Scalar(255));

		cv::Mat XCoeffs, YCoeffs;
		fitLineLine(contours[i], XCoeffs, YCoeffs);
		std::cout << "XCoeffs: " << XCoeffs << std::endl;
		std::cout << "YCoeffs: " << YCoeffs << std::endl;
	}


	//at this point let's get our character contours by line
	std::vector<std::vector<std::vector<cv::Point>>> charsonlines = getCharacterContoursByLine(contours, charcontours);
	std::vector<std::vector<std::vector<cv::Point>>> contourWords = getContourWords(charsonlines);

	std::cout << "Words detected: " << contourWords.size() << "\n";

	//we could even draw those contours just to verify that they work

	//iterate through lines
	for (int i = 0; i < charsonlines.size(); i++)
	{
		std::vector<std::vector<cv::Point>> t_singleContour(1);
		
		//iterate through characters
		for (int j = 0; j < charsonlines[i].size(); j++)
		{
			t_singleContour[0] = charsonlines[i][j];
			cv::drawContours(book, t_singleContour, 0, cv::Scalar(255));
			
		}
	}

	//after figuring out the correction coefficients, let's try to allocate the characters to each line:

	//std::vector<std::vector<std::vector<cv::Point>>> characterLines = getCharacterContoursByLine(contour, )
	
	cv::namedWindow("Text masks window", CV_WINDOW_AUTOSIZE);
	cv::imshow("Text masks window", book);
	return thresholdMasks;
}

/**Uses SDL to draw a character to the canvas. 
*/
void stringToMat(std::string in, cv::Mat &out, TTF_Font *font)
{
	SDL_Color textColor = { 255, 255, 255 };
	
	SDL_Surface *surface = NULL;

	

	if (font == NULL)
	{
		std::cerr << "oh fuck the font didn't... y'know, font." << std::endl;
		return;
	}

	surface = TTF_RenderText_Solid(font, in.c_str(), textColor);
	std::cout << "Surface type: " << static_cast<int>(SDL_PIXELTYPE(surface->format->format)) << std::endl;

	cv::Mat charmat(surface->h, surface->pitch, CV_8UC1);
	

	//just a copy
	for (int i = 0; i < surface->pitch; i++)
	{
		for (int j = 0; j < surface->h; j++)
		{
			unsigned char next = ((unsigned char*)surface->pixels)[(j * surface->pitch + i)]*255;
			std::cout << static_cast<int>(next) << " ";
			charmat.at<unsigned char>(j,  i) = next;
		}
		std::cout << "\n";
	}
	((unsigned char*)surface->pixels)[1] = 0;
	//cv::GaussianBlur(charmat, charmat, cv::Size(7, 7), 2);
	out = charmat;

	SDL_FreeSurface(surface);
}




/*Generates a texture full of grey noise*/

cv::Mat noise(cv::Size size, float noisiness)
{
	std::exception("Attempted to use unimplemented function noise!");
	cv::Mat noiseMatrix(size, CV_8UC1);
	for (int i = 0; i < size.width; i++)
	{
		for (int j = 0; j < size.height; j++)
		{
			//first generate a random number;
		}
	}
	return noiseMatrix;
}

void getContoursFromLinesOrSomething(cv::Mat bookImage, std::vector<std::vector<cv::Point>> &lines)
{
	cv::threshold(bookImage, bookImage, 0, 255, CV_THRESH_OTSU | CV_THRESH_BINARY);
	//invert the image

	cv::Mat inv = cv::Mat::ones(bookImage.rows, bookImage.cols, bookImage.type()) * 255 - bookImage;

	std::vector<cv::Vec4i> hierarchy;
	std::vector<std::vector<cv::Point>> contours;

	cv::GaussianBlur(inv, inv, cv::Size(1, 13), 3);

	cv::findContours(inv, lines, hierarchy, CV_RETR_LIST, CV_CHAIN_APPROX_NONE);

}


//most basicist test of opencv ever
int main(int argc, char** argv)
{
	SDL_Init(SDL_INIT_EVERYTHING);
	TTF_Init();
	TTF_Font* myfont = NULL;
	myfont = TTF_OpenFont(fonts[2], 30);


	if (!myfont)
	{
		exit(1);
	}

	//just for funsies:

	cv::Mat charmat;

	stringToMat("Poorly photocopied text filter", charmat, myfont);


	cv::Mat image = cv::imread("text.png", CV_LOAD_IMAGE_GRAYSCALE);
	std::vector<std::vector<cv::Point>> contours;
	cv::vector<cv::Vec4i> hierarchy;
	cv::Mat thresh_out;
	cv::Mat inv_out = cv::Mat::ones(cv::Size(image.size()), CV_8UC1) * 255 - image;
	//now blur - first pass we do a heavily biased vertical 'mostly' gaussian blur to get things like the dots on the 'i' character to register as the same character

	cv::GaussianBlur(inv_out, inv_out, cv::Size(3, 31), 4);

	cv::threshold(inv_out, thresh_out, 0, 255, CV_THRESH_OTSU | CV_THRESH_BINARY);
	cv::findContours(thresh_out, contours, hierarchy , CV_RETR_LIST, CV_CHAIN_APPROX_SIMPLE, cv::Point(0,0));
	std::map <int, char> charmap;

	std::stringstream predicted;
	std::vector<int> sizes;
	for (auto it = contours.begin(); it != contours.end(); it++)
	{
		int size = (*it).size(); 
		cv::Rect myrect = cv::boundingRect(*it);

		sizes.push_back(size);

		std::stringstream stream;
		stream << size;


		cv::putText(thresh_out, stream.str(), myrect.tl(), CV_FONT_HERSHEY_SIMPLEX, 0.3, cv::Scalar(255));
	}

	auto blur = [](cv::Mat &tomod) { cv::GaussianBlur(tomod, tomod, cv::Size(3, 3), 1); };
	auto noise = [](cv::Mat &tomod) {
		cv::Mat tmat = cv::Mat(tomod.rows, tomod.cols, tomod.type());
		cv::randn(tmat, 64, 64);
		//since we're adding noise...
		tomod -= tmat;
	};

	auto smoothnoise = [&blur](cv::Mat &tomod) {
		//first generate some noise
		cv::Mat tmat = cv::Mat(tomod.rows, tomod.cols, tomod.type());
		cv::randn(tmat, 64, 64);
		//blur the noise
		blur(tmat);

		tomod = tmat/2 + tomod/2;



	};

	std::cout << predicted.str();
	

	//cv::GaussianBlur(charmat, charmat, cv::Size(5, 1), 3);
	//smoothnoise(charmat);
	cv::Mat x;
	charmat.copyTo(x);
	noise(charmat);
	blur(charmat);
	cv::threshold(charmat, charmat, 0, 255, CV_THRESH_OTSU);
	charmat = cv::Mat::ones(cv::Size(charmat.size()), CV_8UC1) * 255 - charmat;
	cv::namedWindow("Training window", CV_WINDOW_AUTOSIZE);
	cv::imshow("Training window", charmat);

	cv::namedWindow("Contour window", CV_WINDOW_AUTOSIZE);
	cv::imshow("Contour window", thresh_out);


	cv::Mat bookmat = cv::imread("book.jpg", CV_LOAD_IMAGE_GRAYSCALE);
	processBookScan(bookmat);


	cv::waitKey(0);
	for (auto it = sizes.begin(); it != sizes.end(); it++)
	{
		int size = *it;
		if (charmap.find(size) == charmap.end())
		{
			//didn't find the size in the map
			char nextchar = ' ';
			std::cout << "Size: " << size << " :: ";
			std::cin >> nextchar;
			charmap[size] = nextchar;
		}

		predicted << charmap[size];
	}

	


	
	return 0;
}