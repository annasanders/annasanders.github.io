---
layout: page
title: ⌨️ About This Website
permalink: /about_website
---
Want to know how I built this website with no web experience? Keep reading!

Already built your website and want to know how I have a custom syntax highlighter? Click [**here**](#pygments)

## 🔍 So You Want To Build a Website
The world is on the internet, so why not have your own website! [Github Pages](https://pages.github.com) is a great way to build a website for free! The process is a bit more complex than other website options, but it's free and is great for displaying code. There are also numerous guides available to help walk through the steps (or just keep reading and I'll try to 😀).

## 🛠️ What You Need
Technically, the only thing *necessary* to create a live website on [**Github Pages**](https://pages.github.com) is a **Github** account; however, any updates made will be done on the live **Github** website. Since **Github Pages** uses [**Jekyll**](https://jekyllrb.com), you can download **Jekyll** (through [**Ruby**](https://www.ruby-lang.org/en/)) and 'run' a copy of your website on your local computer. This will display the copy of your website in a browser like you would see it on the web. 

A text editor is also advised. Whether you're using HTML or Markdown, all relevant files are fully editable in text. I personally use [Notepad++](https://notepad-plus-plus.org), which is available on Windows.

I would also recommend downloading the base **Github Pages** theme, [minima](https://github.com/jekyll/minima), a **Jekyll** theme, which is downloadable as a 'gem' in **Ruby**. Downloading the gem files and copying them to your Github/Jekyll site files can allow for more customization options. 

## 🚂 Getting Started 
1. Install [**Ruby**](https://www.ruby-lang.org/en/downloads/)
1. If you haven't already, download [**Github for desktop**](https://desktop.github.com)
1. Create your [**Github Pages Website**](https://docs.github.com/en/pages/getting-started-with-github-pages/creating-a-github-pages-site) and pull the repository to your computer
1. Install [**Jekyll**](https://jekyllrb.com/docs/) and create your 'site' (myblog example instructions)
1. Copy the files created in your **Github** repository into your created website or change the **Jekyll** directory to the **Github** website repository
1. Run **Jekyll** ('Jekyll Serve' command in **Ruby**) and view your website!

## 😶 Now What
Well, now you have a website!

### Interacting With Your Website
If you're using **Jekyll** on your local computer, editing and adding pages to your website should be relativly instantaneous, so long as you save your file and reload the page. If you want to make changes to your live website, you'll need to edit the files on your local computer's **Github** repository and push the changes. The updates may take a few minutes to process.

Keep in mind, editing files will create **.bak** files, which are copies of the previous version of the file. These will not affect your website, but are useful if you accidently deleted something important.

### Home Page and Page Types
The **index.markdown** file is the first page on your website, and should be what you see when you visit the **'yourwebsitehere'.github.io** website link. 

If you view the file, you should see something like this at the top (this is what's currently in my index file):
```markdown
---
# Feel free to add content and custom Front Matter to this file.
# To modify the layout, see https://jekyllrb.com/docs/themes/#overriding-theme-defaults

layout: home
title: 🦕 Welcome to my Website!
---
```

All markdown pages will have a header similar to this. The 'layout' sections are defined in the **_layouts** folder (note: you may need to copy the the minima theme into your site files if not available). There should be a **default.html**, **home.html**, **page.html**, and **post.html** file. These will format your website based on the 'layout' defined in the header. 

Minima has blogging capabilities, where it will display blog posts on your home page. I am not currently using this feature, so I removed the blog page navigation and do not use 'post' page layouts. 

### _config.yml
The **_config.yml** stores most of the website configurations, including webiste title, website theme, headers (e.g. 'R Projects', 'Python Projects', 'Other Projects'), social media links, and url settings (these can be blank if you're using **Github Pages**.

Check out the great [minima **Github** documentation](https://github.com/jekyll/minima) for a full guide on how you can customize your website! 

Don't like minima? Try [another supported **Github** themes](https://pages.github.com/themes/)!

### Expanding and Managing Your Growing Website
To create a new website page, you can add a Markdown or HTML file, or copy an existing file to modify. To link to another website page, you will need to reference **{\{ site.url }}/page**, which will find the file in the site directory with the permalink of **'/page'**. Take this page header:
```markdown
---
layout: page
title: ⌨️ About This Website
permalink: /about_website
---
```

**{\{ site.url }}** will populate in your site url; for locally hosted, it should be **'localhost:####'**, for your live website, it should be your **'yourwebsitehere'.github.io** website link. You can get to this page above by linking **{\{ site.url }}/about_website**, which will be the same as: **https://annasanders.github.io/ about_website** on the web.

Having a website with many different pages can quickly create confusing site files. So long as you have the correct website path direct links, you can create as many internal file folders needed! This can be helpful if you save or reference lots of static images. Linking plots is as easy as: \[](**{\{ site.url }}/plots/plot1.jpg**).


## Advanced Customizations
There are workarounds to customize site settings without changing to another theme. My website still uses minima, but I have a custom green accent color, as well as custom syntax and code highlighting.
s
### Modifying minima _sass
After copying the minima files into your site files, you should see a **_sass** folder. Depending on your minima version, you may have a **.scss** named **minima.scss** or **classic.scss**. This is the default minima style configurations. Take for example a section of my **minima.css**:
```scss
$text-color:       #111 !default;
$background-color: #F8F8F8 !default;
$brand-color:      #7BB517 !default;
```

The **$brand-color:** has been changed to #7BB517 (hexadecimal color), which is a mid-dark green color. Now, any calls to the **$brand-color** will default in this color, instead of the blue color.

### Modifying minima _base.scss
Like the **minima.scss** file, the **_base.scss** file should be in the same location and also controls minima style configurations. Take a look at this section of my **_base.scss** file:

```scss
/**
 * Links
 */
a {
  color: $brand-color;
  text-decoration: none;

  &:visited {
    color: darken($brand-color, 15%);
  }

  &:hover {
    color: $text-color;
    text-decoration: underline;
  }

  .social-media-list &:hover {
    text-decoration: none;

    .username {
      text-decoration: underline;
      text-decoration: underline;
    }
  }
}
```
As you can see, the [links](https://www.youtube.com/watch?v=dQw4w9WgXcQ) on my page should display in the #7BB517 green color I specified in the **minima.scss** and are automatically underlined and darkened if visited. 

This code chunk allows me to change the background color of my code blocks (\```\[language] in markdown):

```scss
/**
 * Code formatting
 */
pre,
code {
  @include relative-font-size(0.9375);
  border: 0px solid $grey-color-light;
  border-radius: 0px;
  background-color: #2A333C;
}

code {
  padding: 1px 0px;
  text-color: #2A333C;
}

pre {
  padding: 8px 12px;
  overflow-x: auto;#44525F

  > code {
    border: 0;
    padding-right: 0;
    padding-left: 0;
  }
}
```

The background-color, #2A333C, is a dark grey-blue, and should be the background color on code chunks throughout my website.

### Pygments
[Pygments](https://pygments.org) is a python syntax highlighter for code that can be used with **Jekyll** and minima. There are some base themes with Pygments, as well as many user created themes out there. [This Github](https://stylishthemes.github.io/Syntax-Themes/pygments/) has a great resource to view some Pygments themes! 

A Pygments syntax highlighter specification file (**.css**) can be hard-coded in **_layouts** files in HTML. Take a look at my **default.html** file in my website:
```html
<!DOCTYPE html>
<html lang="{{ page.lang | default: site.lang | default: "en" }}">
<link href="/css/pygments.css" rel="stylesheet">
<!-- Website Code
deleted due to code chunk defaulting in another website page
-->
  </body>
</html>
``` 

The href location is my **pygments.css** file in my website. My **pygments.css** file is based off of [Base16 Default Dark](https://github.com/idleberg) with various modifications has:
```css
/*! Base16 Default Dark by Chris Kempson; https://github.com/idleberg */
.highlight .c { color: #BDC9A6 } /* Comment */
.highlight .err { color: #ac4142 } /* Error */
.highlight .k { color: #aa759f } /* Keyword */
.highlight .l { color: #d28445 } /* Literal */
.highlight .n, .highlight .h { color: #e8e8e8 } /* Name */
.highlight .o { color: #75b5aa } /* Operator */
.highlight .p { color: #e8e8e8 } /* Punctuation */
.highlight .cm { color: #BDC9A6 } /* Comment.Multiline */
.highlight .cp { color: #BDC9A6 } /* Comment.Preproc */
.highlight .c1 { color: #BDC9A6 } /* Comment.Single */
.highlight .cs { color: #BDC9A6 } /* Comment.Special */
.highlight .gd { color: #ac4142 } /* Generic.Deleted */
.highlight .ge { font-style: italic } /* Generic.Emph */
.highlight .gh { color: #e8e8e8; font-weight: bold } /* Generic.Heading */
.highlight .gi { color: #90a959 } /* Generic.Inserted */
.highlight .gp { color: #e8e8e8; font-weight: bold } /* Generic.Prompt */
.highlight .gs { color: #e8e8e8; font-weight: bold } /* Generic.Strong */
.highlight .gu { color: #75b5aa; font-weight: bold } /* Generic.Subheading */
.highlight .kc { color: #aa759f } /* Keyword.Constant */
.highlight .kd { color: #aa759f } /* Keyword.Declaration */
.highlight .kn { color: #75b5aa } /* Keyword.Namespace */
.highlight .kp { color: #aa759f } /* Keyword.Pseudo */
.highlight .kr { color: #aa759f } /* Keyword.Reserved */
.highlight .kt { color: #f4bf75 } /* Keyword.Type */
.highlight .ld { color: #90a959 } /* Literal.Date */
.highlight .m { color: #d28445 } /* Literal.Number */
.highlight .s { color: #90a959 } /* Literal.String */
.highlight .na { color: #6a9fb5 } /* Name.Attribute */
.highlight .nb { color: #e8e8e8 } /* Name.Builtin */
.highlight .nc { color: #f4bf75 } /* Name.Class */
.highlight .no { color: #ac4142 } /* Name.Constant */
.highlight .nd { color: #75b5aa } /* Name.Decorator */
.highlight .ni { color: #e8e8e8 } /* Name.Entity */
.highlight .ne { color: #ac4142 } /* Name.Exception */
.highlight .nf { color: #6a9fb5 } /* Name.Function */
.highlight .nl { color: #e8e8e8 } /* Name.Label */
.highlight .nn { color: #f4bf75 } /* Name.Namespace */
.highlight .nx { color: #6a9fb5 } /* Name.Other */
.highlight .py { color: #e8e8e8 } /* Name.Property */
.highlight .nt { color: #75b5aa } /* Name.Tag */
.highlight .nv { color: #ac4142 } /* Name.Variable */
.highlight .ow { color: #75b5aa } /* Operator.Word */
.highlight .w { color: #e8e8e8 } /* Text.Whitespace */
.highlight .mf { color: #d28445 } /* Literal.Number.Float */
.highlight .mh { color: #d28445 } /* Literal.Number.Hex */
.highlight .mi { color: #d28445 } /* Literal.Number.Integer */
.highlight .mo { color: #d28445 } /* Literal.Number.Oct */
.highlight .sb { color: #90a959 } /* Literal.String.Backtick */
.highlight .sc { color: #e8e8e8 } /* Literal.String.Char */
.highlight .sd { color: #505050 } /* Literal.String.Doc */
.highlight .s2 { color: #90a959 } /* Literal.String.Double */
.highlight .se { color: #d28445 } /* Literal.String.Escape */
.highlight .sh { color: #90a959 } /* Literal.String.Heredoc */
.highlight .si { color: #d28445 } /* Literal.String.Interpol */
.highlight .sx { color: #90a959 } /* Literal.String.Other */
.highlight .sr { color: #90a959 } /* Literal.String.Regex */
.highlight .s1 { color: #90a959 } /* Literal.String.Single */
.highlight .ss { color: #90a959 } /* Literal.String.Symbol */
.highlight .bp { color: #e8e8e8 } /* Name.Builtin.Pseudo */
.highlight .vc { color: #ac4142 } /* Name.Variable.Class */
.highlight .vg { color: #ac4142 } /* Name.Variable.Global */
.highlight .vi { color: #ac4142 } /* Name.Variable.Instance */
.highlight .il { color: #d28445 } /* Literal.Number.Integer.Long */
.highlight { color: #e8e8e8 }  /* Everything Else*/
```
If you're having trouble figuring out what portion of your code chunk is highlighting, try Inspecting your website (either your local site or your actual website should work). Within the code block, the website should break out individual \<code\> into <span class=''> for each element Pygments can find based on the coding language used. For the **_base.css** code chunk [above](#modifying-minima-_basescss), Pygments was able to find the multi-line comment ("Links") and highlight it to the .cm color:
```html
<span class="cm">/** * Links */</span>
```

Inspecting Elements can also be helpful to directly link to page sections. The 'above' link in the previous section is:
```markdown
[above](#modifying-minima-_basescss)
```
Which is the HTML element:
```html
<h4 id="modifying-minima-basescss">Modifying minima _base.scss</h4>
```

**Note:** you may need to do some extra website style fiddling to stop minima from forcing in its default syntax highlighting. 

## Other Words of Advice
Rome wasn't built in a day, so don't worry if your website isn't either! I'm constantly adding to my website too 🙃.