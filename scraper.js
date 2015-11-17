var request = require('request');
var cheerio = require('cheerio');
var fs = require('fs');
var webpage = 'http://votesmart.org/candidate/key-votes/31969/tom-graves-jr?p=';
var end = 11;
var filename = "tgraves.csv";
// var webpage = process.argv[2];
// var end = process.argv[3];
// var filename = process.argv[4];
//USE PROMISES?
function scraper(webpage, end, filename){
	var wstream = fs.createWriteStream(filename);
	for(i = 1; i < end+1; i++){
		var webpg = webpage + i
		request(webpg, function (error, response, body) {
			var towrite = "";
  			if (!error && response.statusCode == 200) {
    			var page = cheerio.load(body);
    			page('table tr').each(function(tr_index, tr){
    				row = "";
    				page(this).find("td").each(function(td_index, td){
    					// console.log(page(this).text());
    					row = row + page(this).text()+"\t";
    				});
    				//Remove line breaks and replace with spaces
    				row = row.replace(/(\r\n|\n|\r)/gm," ");
    				// console.log(row);
    				// console.log("\n");
    				if(row.match(/[a-z0-9]/i)){
    					wstream.write(row + "\n");
    				}
    			});
  			}else{ 
  				console.log("ERROR");
  			}
		});
	}
}
scraper(webpage, end);


